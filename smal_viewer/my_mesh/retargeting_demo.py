import argparse
import numpy as np
import trimesh
import os
import pickle
from scipy.spatial import KDTree, procrustes, distance
from scipy.optimize import minimize
from pathlib import Path
import sys

# Add the parent directory to sys.path to find the models module
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Try to import SMPL package
try:
    from SMPL.smpl_webuser.serialization import load_model
    HAS_SMPL = True
except ImportError:
    HAS_SMPL = False
    print("Warning: SMPL package not found. Falling back to native implementation.")

# Import our wrapper and model
try:
    from smal_viewer.models.smpl_wrapper import SMPLWrapper
    from smal_viewer.models.smal_model import SMALModel
except ImportError as e:
    print(f"Error importing SMAL/SMPL models: {e}")
    print("Please ensure the smal_viewer package is correctly structured and in the Python path.")
    sys.exit(1)

def load_obj(file_path):
    """Loads vertices and faces from an OBJ file."""
    try:
        mesh = trimesh.load(file_path, process=False)
        return mesh.vertices, mesh.faces
    except Exception as e:
        print(f"Error loading OBJ file {file_path}: {e}")
        # Fallback simple parsing
        vertices = []
        faces = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        vertices.append(list(map(float, line.strip().split()[1:])))
                    elif line.startswith('f '):
                        # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                        face = [int(v.split('/')[0]) - 1 for v in line.strip().split()[1:]]
                        faces.append(face)
            return np.array(vertices), np.array(faces)
        except Exception as fallback_e:
            print(f"Fallback OBJ parsing failed: {fallback_e}")
            return None, None


def save_obj(file_path, vertices, faces):
    """Saves vertices and faces to an OBJ file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(file_path)
        print(f"Saved retargeted mesh to {file_path}")
    except Exception as e:
        print(f"Error saving OBJ using trimesh: {e}. Trying manual save.")
        try:
            with open(file_path, 'w') as f:
                for v in vertices:
                    f.write(f"v {' '.join(map(str, v))}\n")
                for face in faces:
                    # OBJ faces are 1-indexed
                    f.write(f"f {' '.join(map(lambda x: str(x + 1), face))}\n")
            print(f"Saved retargeted mesh manually to {file_path}")
        except Exception as manual_e:
            print(f"Manual OBJ saving failed: {manual_e}")


def find_smal_model(base_path='.'):
    """Attempts to find the SMAL model file."""
    possible_paths = [
        Path(base_path) / "data/smal_CVPR2017.pkl",
        Path(base_path) / "smal_CVPR2017.pkl",
        project_root / "data/smal_CVPR2017.pkl",
    ]
    for path in possible_paths:
        if path.exists():
            print(f"Found SMAL model at: {path}")
            return str(path)
    return None


def compute_chamfer_distance(source_points, target_points):
    """
    Compute the Chamfer distance between two point clouds.
    
    Args:
        source_points: Source point cloud (Nx3)
        target_points: Target point cloud (Mx3)
        
    Returns:
        float: Chamfer distance
    """
    # For each point in source, find the nearest point in target
    kdtree_target = KDTree(target_points)
    dist_source_to_target, _ = kdtree_target.query(source_points)
    
    # For each point in target, find the nearest point in source
    kdtree_source = KDTree(source_points)
    dist_target_to_source, _ = kdtree_source.query(target_points)
    
    # Calculate average distance in both directions
    chamfer_dist = np.mean(dist_source_to_target) + np.mean(dist_target_to_source)
    return chamfer_dist


def rigid_align_with_scaling(source_points, target_points, use_icp=True, max_iterations=20):
    """
    Perform rigid alignment with non-uniform scaling between point clouds.
    This is more robust than simple rigid alignment for meshes with different proportions.
    
    Args:
        source_points: Source point cloud (Nx3)
        target_points: Target point cloud (Mx3)
        use_icp: Whether to refine alignment with ICP
        max_iterations: Maximum number of ICP iterations
        
    Returns:
        tuple: (transform_matrix, aligned_source) - 4x4 transform matrix and aligned source points
    """
    print(f"Performing robust alignment with non-uniform scaling and ICP...")
    print(f"Source points shape: {source_points.shape}, Target points shape: {target_points.shape}")
    
    # Center the point clouds
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid
    
    # Compute scaling factors for each axis
    source_scale = np.max(np.abs(centered_source), axis=0)
    target_scale = np.max(np.abs(centered_target), axis=0)
    
    # Avoid division by zero
    source_scale = np.maximum(source_scale, 1e-10)
    
    # Compute scaling transformation (non-uniform)
    scaling = target_scale / source_scale
    print(f"Non-uniform scaling factors: {scaling}")
    
    # Apply scaling to centered source
    scaled_source = centered_source * scaling
    
    # Initialize rotation as identity
    R = np.eye(3)
    
    # If using ICP, refine the alignment iteratively
    if use_icp:
        print("Refining alignment with ICP...")
        kdtree = KDTree(centered_target)
        
        for iteration in range(max_iterations):
            # Find closest points
            _, indices = kdtree.query(scaled_source)
            
            # Get corresponding points
            corresponding_points = centered_target[indices]
            
            # Compute transformation using SVD
            H = np.dot(scaled_source.T, corresponding_points)
            U, _, Vt = np.linalg.svd(H)
            
            # Compute rotation that aligns the points
            R_new = np.dot(Vt.T, U.T)
            
            # Ensure it's a proper rotation matrix (det=1)
            if np.linalg.det(R_new) < 0:
                Vt[-1, :] *= -1
                R_new = np.dot(Vt.T, U.T)
            
            # Apply rotation
            scaled_source = np.dot(scaled_source, R_new)
            
            # Update cumulative rotation
            R = np.dot(R_new, R)
            
            # Calculate error
            error = np.mean(np.sum((scaled_source - corresponding_points)**2, axis=1))
            print(f"ICP iteration {iteration+1}/{max_iterations}, error: {error:.6f}")
            
            # Check for convergence
            if error < 1e-5:
                print("ICP converged early")
                break
    
    # Assemble the full transformation matrix (4x4)
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scaling) @ R
    transform[:3, 3] = target_centroid - np.dot(source_centroid, transform[:3, :3])
    
    # Apply transformation to original source points
    aligned_source = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    aligned_source = np.dot(aligned_source, transform.T)[:, :3]
    
    # Calculate final alignment quality
    chamfer_dist = compute_chamfer_distance(aligned_source, target_points)
    print(f"Final alignment Chamfer distance: {chamfer_dist}")
    
    return transform, aligned_source


def compute_weights_matrix(source_points, target_points, indices=None, use_rbf=True):
    """
    Compute the weights matrix for interpolation between source and target meshes.
    
    This uses either barycentric coordinates (for each vertex in source find the 
    nearest triangle in target) or Radial Basis Functions (for smoother interpolation).
    
    Args:
        source_points: Source vertices (Nx3)
        target_points: Target vertices (Mx3)
        indices: Nearest neighbor indices from target to source (optional)
        use_rbf: Whether to use RBF interpolation (default: True)
        
    Returns:
        numpy.ndarray: Weights matrix (NxM) where W[i,j] is the influence of target vertex j on source vertex i
    """
    n_source = source_points.shape[0]
    n_target = target_points.shape[0]
    
    # Find nearest neighbors if indices not provided
    if indices is None:
        print("Building KD-tree for nearest neighbor search...")
        kdtree = KDTree(target_points)
        print("Finding nearest neighbors...")
        k_neighbors = min(4, n_target)  # Make sure k isn't larger than available points
        _, indices = kdtree.query(source_points, k=k_neighbors)
    
    if use_rbf:
        print("Computing RBF weights...")
        # Use RBF for smooth interpolation
        weights = np.zeros((n_source, n_target))
        
        # Use a simple Gaussian RBF
        # Dynamically set sigma based on model scale (approx. 1% of model size)
        model_size = np.max(np.ptp(target_points, axis=0))
        sigma = model_size * 0.01
        print(f"Using RBF with sigma={sigma} (scaled to model size)")
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for batch_start in range(0, n_source, batch_size):
            batch_end = min(batch_start + batch_size, n_source)
            print(f"Processing vertices {batch_start} to {batch_end}...")
            
            for i in range(batch_start, batch_end):
                # Get distances to nearest neighbors only (to save memory)
                nn_indices = indices[i]
                nn_points = target_points[nn_indices]
                dists = np.sum((nn_points - source_points[i])**2, axis=1)
                
                # Apply RBF kernel
                weights_i = np.exp(-dists / (2 * sigma**2))
                
                # Handle normalization safely
                sum_weights = np.sum(weights_i)
                if sum_weights > 1e-10:  # If sum is not too close to zero
                    weights_i = weights_i / sum_weights
                else:
                    # If all weights are near zero, just use the nearest point
                    weights_i = np.zeros_like(weights_i)
                    weights_i[np.argmin(dists)] = 1.0
                
                # Store in weight matrix (sparse representation)
                for j, idx in enumerate(nn_indices):
                    if weights_i[j] > 1e-6:  # Skip very small weights
                        weights[i, idx] = weights_i[j]
            
        return weights
    else:
        print("Computing inverse distance weights...")
        # Simpler approach: use fixed nearest neighbors with distance-based weights
        weights = np.zeros((n_source, n_target))
        
        # Process in batches
        batch_size = 1000
        for batch_start in range(0, n_source, batch_size):
            batch_end = min(batch_start + batch_size, n_source)
            print(f"Processing vertices {batch_start} to {batch_end}...")
            
            for i in range(batch_start, batch_end):
                nn_indices = indices[i]
                
                # Get distances to nearest neighbors
                nn_points = target_points[nn_indices]
                dists = np.sum((nn_points - source_points[i])**2, axis=1)
                
                # Inverse distance weighting
                if np.min(dists) < 1e-10:
                    # If we have an exact match, use that point only
                    w = np.zeros(len(nn_indices))
                    w[np.argmin(dists)] = 1.0
                else:
                    # Otherwise use inverse distance weights
                    w = 1.0 / (dists + 1e-10)
                    sum_w = np.sum(w)
                    if sum_w > 1e-10:
                        w = w / sum_w  # Normalize
                    else:
                        # If all weights are near zero, just use the nearest point
                        w = np.zeros_like(w)
                        w[np.argmin(dists)] = 1.0
                
                # Set the weights
                for j, idx in enumerate(nn_indices):
                    if w[j] > 1e-6:  # Skip very small weights
                        weights[i, idx] = w[j]
                
        return weights


def main():
    parser = argparse.ArgumentParser(description="Retarget SMAL parameters from a PKL file to a high-quality render mesh.")
    parser.add_argument('--obj', type=str, required=True, help='Path to the high-quality render mesh template (.obj)')
    parser.add_argument('--params_pkl', type=str, required=True, help='Path to the .pkl file containing target \'pose\' and \'betas\' parameters.')
    parser.add_argument('--template_pkl', type=str, default="data/cows/cow_alph4.pkl", help='Path to the PKL file to use as template parameters')
    parser.add_argument('--smal_model_path', type=str, default=None, help='Path to the SMAL model file (.pkl)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the retargeted mesh')
    parser.add_argument('--use_rbf', action='store_true', help='Use RBF for smoother interpolation')
    parser.add_argument('--save_templates', action='store_true', default=True, help='Save the template and aligned SMAL meshes as OBJ files')
    parser.add_argument('--use_smpl', action='store_true', default=True, help='Use SMPL package for model evaluation')
    
    args = parser.parse_args()

    # --- 1. Load Models & Params ---
    smal_model_path = args.smal_model_path or find_smal_model()
    if not smal_model_path:
        print("Error: SMAL model file (.pkl) not found. Please specify with --smal_model_path.")
        sys.exit(1)

    if not Path(args.obj).exists():
         print(f"Error: Render geometry file not found at {args.obj}")
         sys.exit(1)

    if not Path(args.params_pkl).exists():
         print(f"Error: Parameter PKL file not found at {args.params_pkl}")
         sys.exit(1)
    
    if not Path(args.template_pkl).exists():
         print(f"Error: Template PKL file not found at {args.template_pkl}")
         sys.exit(1)

    # Load target parameters from PKL file
    print(f"Loading target parameters from {args.params_pkl}...")
    try:
        with open(args.params_pkl, 'rb') as f:
            params_data = pickle.load(f, encoding='latin1')

            # Check for common key variations
            if 'pose' in params_data:
                pose_target = np.array(params_data['pose']).flatten()
            elif 'theta' in params_data:
                 pose_target = np.array(params_data['theta']).flatten()
            else:
                raise KeyError("Could not find 'pose' or 'theta' key in target PKL file.")

            if 'betas' in params_data:
                 betas_target = np.array(params_data['betas']).flatten()
            elif 'beta' in params_data:
                 betas_target = np.array(params_data['beta']).flatten()
            elif 'shape' in params_data:
                 betas_target = np.array(params_data['shape']).flatten()
            else:
                 raise KeyError("Could not find 'betas', 'beta', or 'shape' key in target PKL file.")

        print(f"Loaded target pose with shape: {pose_target.shape}")
        print(f"Loaded target betas with shape: {betas_target.shape}")
    except Exception as e:
        print(f"Error loading or parsing target PKL file {args.params_pkl}: {e}")
        sys.exit(1)
    
    # Load template parameters from specified PKL file
    print(f"Loading template parameters from {args.template_pkl}...")
    try:
        with open(args.template_pkl, 'rb') as f:
            template_data = pickle.load(f, encoding='latin1')

            # Check for common key variations
            if 'pose' in template_data:
                pose_template = np.array(template_data['pose']).flatten()
            elif 'theta' in template_data:
                 pose_template = np.array(template_data['theta']).flatten()
            else:
                raise KeyError("Could not find 'pose' or 'theta' key in template PKL file.")

            if 'betas' in template_data:
                 betas_template = np.array(template_data['betas']).flatten()
            elif 'beta' in template_data:
                 betas_template = np.array(template_data['beta']).flatten()
            elif 'shape' in template_data:
                 betas_template = np.array(template_data['shape']).flatten()
            else:
                 raise KeyError("Could not find 'betas', 'beta', or 'shape' key in template PKL file.")

        print(f"Loaded template pose with shape: {pose_template.shape}")
        print(f"Loaded template betas with shape: {betas_template.shape}")
    except Exception as e:
        print(f"Error loading or parsing template PKL file {args.template_pkl}: {e}")
        sys.exit(1)

    # Determine which model evaluation method to use
    use_smpl = args.use_smpl and HAS_SMPL
    
    print("Loading SMAL model...")
    if use_smpl:
        print("Using SMPL package for model evaluation")
        # Load the model using SMPL's load_model function
        smpl_model = load_model(smal_model_path)
        
        # For the cow family, load shape parameters from data file
        data_path = "./data/smal_CVPR2017_data.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Set to bovidae (cow) family - index 3
            if 'cluster_means' in data and len(data['cluster_means']) > 3:
                print("Applied bovidae family shape parameters from SMAL data")
        
        # Create a wrapper for the SMPL model to make it compatible with our viewer
        model = SMPLWrapper(smpl_model)
    else:
        # Use our native implementation
        print("Using native implementation for model evaluation")
        # Load SMAL model - always use family 3 (cow)
        model = SMALModel(model_path=smal_model_path, shape_family_id=3)
        
        # Initialize the model specifically for cows
        if hasattr(model, 'initialize_for_cow'):
            model.initialize_for_cow()
        
        # Wrap with SMPLWrapper for consistent interface
        model = SMPLWrapper(model)
    
    print(f"Model loaded with faces shape: {model.f.shape}")

    print(f"Loading render mesh template from {args.obj}...")
    V_render_template, F_render = load_obj(args.obj)
    if V_render_template is None:
        print(f"Failed to load render mesh from {args.obj}")
        sys.exit(1)
    print(f"Render mesh loaded with {V_render_template.shape[0]} vertices.")

    # --- 2. Establish Template SMAL State (using cow_alph4.pkl) ---
    print("Setting model to template parameters from cow_alph4.pkl...")
    try:
        # Apply template parameters
        V_smal_template = model.forward(pose=pose_template, betas=betas_template)
        print(f"Generated template mesh using cow_alph4.pkl with shape: {V_smal_template.shape}")
        print(f"Template betas (first 5): {betas_template[:5]}")

        # Save the template mesh (cow_alph4 parameters) as OBJ
        if args.save_templates:
            template_filename = "smal_template_cow_alph4.obj"
            template_path = os.path.join(args.output_dir, template_filename)
            print(f"Saving template mesh to {template_path}...")
            save_obj(template_path, V_smal_template, model.f)
            print(f"Saved template mesh.")

    except Exception as e:
         print(f"Error generating template mesh: {e}")
         sys.exit(1)

    # --- 3. Perform Enhanced Rigid Alignment between template and Render template ---
    print("Performing enhanced alignment between template mesh and render mesh...")
    try:
        # Use our improved alignment function with non-uniform scaling and ICP
        transform, V_smal_aligned = rigid_align_with_scaling(V_smal_template, V_render_template, use_icp=True)
        print(f"Alignment complete. Transform matrix shape: {transform.shape}")
        
        # Save the transformation parameters for reference
        np.savetxt(os.path.join(args.output_dir, "alignment_transform.txt"), transform)
        
        # Check alignment quality with Chamfer distance
        cd = compute_chamfer_distance(V_smal_aligned, V_render_template)
        print(f"Post-alignment Chamfer distance: {cd}")
        
        # Save the aligned template mesh as OBJ
        if args.save_templates:
            aligned_template_filename = "smal_template_cow_alph4_aligned.obj"
            aligned_template_path = os.path.join(args.output_dir, aligned_template_filename)
            print(f"Saving aligned template mesh to {aligned_template_path}...")
            save_obj(aligned_template_path, V_smal_aligned, model.f)
            print(f"Saved aligned template mesh.")
        
    except Exception as e:
        print(f"Error in rigid alignment: {e}. Proceeding without alignment.")
        transform = np.eye(4)
        V_smal_aligned = V_smal_template

    # --- 4. Compute the interpolation matrix between aligned meshes ---
    # IMPORTANT: Weight matrix maps from aligned SMAL template (cow_alph4 params) to render mesh
    print("Computing interpolation weights matrix between aligned cow_alph4 SMAL and render mesh...")
    try:
        # Build a KD-tree for nearest neighbor search
        kdtree = KDTree(V_smal_aligned)
        
        # Find nearest neighbors of render mesh vertices in the aligned SMAL mesh
        _, indices = kdtree.query(V_render_template, k=1)
        
        # Compute interpolation weights
        # These weights map from the aligned cow_alph4 SMAL mesh to the render mesh
        W = compute_weights_matrix(V_render_template, V_smal_aligned, indices=indices, use_rbf=args.use_rbf)
        print(f"Computed weights matrix with shape: {W.shape}")
        
        # Save the interpolation matrix for future use
        # This is important as the matrix can be reused for different poses/shapes
        matrix_filename = "interpolation_matrix_cow_alph4.npz"
        matrix_path = os.path.join(args.output_dir, matrix_filename)
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        try:
            np.savez_compressed(
                matrix_path, 
                W=W, 
                indices=indices, 
                transform=transform,
                template_pkl=args.template_pkl
            )
            print(f"Saved interpolation matrix to {matrix_path}")
        except Exception as e:
            print(f"Warning: Could not save interpolation matrix: {e}")
        
    except Exception as e:
        print(f"Error computing interpolation weights: {e}. Using nearest-neighbor interpolation.")
        # Fallback to simple nearest neighbor
        kdtree = KDTree(V_smal_aligned)
        _, indices = kdtree.query(V_render_template, k=1)
        # Create a one-hot weight matrix based on nearest neighbors
        W = np.zeros((V_render_template.shape[0], V_smal_aligned.shape[0]))
        for i, idx in enumerate(indices):
            W[i, idx] = 1.0

    # --- 5. Generate Target Mesh using the loaded parameters ---
    print("Generating target mesh state using parameters from target PKL...")
    try:
        # Generate vertices for the target state using loaded params
        V_smal_target = model.forward(pose=pose_target, betas=betas_target)
        print(f"Generated target mesh with shape: {V_smal_target.shape}")
        
        # Apply the same transformation to the target mesh
        # Convert to homogeneous coordinates (add column of ones)
        target_homogeneous = np.hstack((V_smal_target, np.ones((V_smal_target.shape[0], 1))))
        V_smal_target_aligned = np.dot(target_homogeneous, transform.T)[:, :3]
        print(f"Aligned target mesh with template using same transformation.")
        
        # Save the target mesh as OBJ
        if args.save_templates:
            target_filename = f"smal_target_{Path(args.params_pkl).stem}.obj"
            target_path = os.path.join(args.output_dir, target_filename)
            print(f"Saving target mesh to {target_path}...")
            save_obj(target_path, V_smal_target, model.f)
            print(f"Saved target mesh.")
            
            aligned_target_filename = f"smal_target_{Path(args.params_pkl).stem}_aligned.obj"
            aligned_target_path = os.path.join(args.output_dir, aligned_target_filename)
            print(f"Saving aligned target mesh to {aligned_target_path}...")
            save_obj(aligned_target_path, V_smal_target_aligned, model.f)
            print(f"Saved aligned target mesh.")
        
    except Exception as e:
        print(f"Error generating target mesh: {e}")
        sys.exit(1)

    # --- 6. Apply the deformation to the render mesh using the interpolation matrix ---
    print("Applying deformation to the render mesh...")
    try:
        # Calculate the deformation of the SMAL mesh
        smal_deformation = V_smal_target_aligned - V_smal_aligned
        
        # Apply the deformation to the render mesh using the weight matrix
        # Each vertex in the render mesh is influenced by a weighted combination of SMAL vertices
        V_render_deformed = V_render_template.copy()
        
        print("Applying weighted deformations to render mesh...")
        # Process in batches to avoid memory issues
        batch_size = 1000
        for batch_start in range(0, V_render_template.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, V_render_template.shape[0])
            print(f"Processing vertices {batch_start} to {batch_end}...")
            
            for i in range(batch_start, batch_end):
                # Get the weights for this vertex
                weights = W[i]
                
                # Find non-zero weights for efficiency
                nonzero_indices = np.nonzero(weights)[0]
                
                if len(nonzero_indices) > 0:
                    # Apply weighted deformation from influential SMAL vertices only
                    deformation = np.zeros(3)
                    for j in nonzero_indices:
                        deformation += weights[j] * smal_deformation[j]
                    
                    # Apply the deformation
                    V_render_deformed[i] = V_render_template[i] + deformation
        
        print(f"Applied deformation to render mesh.")
        
    except Exception as e:
        print(f"Error applying deformation: {e}. Using nearest-neighbor deformation instead.")
        # Fallback to simple nearest neighbor deformation
        for i in range(V_render_template.shape[0]):
            nn_idx = indices[i]
            V_render_deformed[i] = V_render_template[i] + smal_deformation[nn_idx]

    # --- 7. Save Output ---
    # Generate filename based on input pkl
    pkl_filename = Path(args.params_pkl).stem
    output_filename = f"retargeted_{pkl_filename}.obj"
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving retargeted render mesh to {output_path}...")
    save_obj(output_path, V_render_deformed, F_render)

    print("Retargeting process completed successfully.")

if __name__ == "__main__":
    main() 