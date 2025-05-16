import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian

# Assuming smal_viewer is in PYTHONPATH or sibling directory
from smal_viewer.utils.loader import load_model_from_pkl # Keep for loading pkl params
from smal_viewer.models.smpl_wrapper import SMPLWrapper

try:
    from SMPL.smpl_webuser.serialization import load_model as smpl_load_model
    # Attempt to import chumpy to check if fake_lib's chumpy is being used, or a real one
    try:
        import chumpy as ch
        print(f"Chumpy imported: {ch.__name__} from {ch.__file__}")
        # More robust check for actual Chumpy vs. a minimal replacement
        IS_CHUMPY_IMPORTED_AND_USABLE = hasattr(ch, 'Ch') and callable(ch.Ch)
    except ImportError:
        IS_CHUMPY_IMPORTED_AND_USABLE = False
        print("Warning: Chumpy could not be imported. fake_lib might not function as expected.")
    HAS_SMPL_SERIALIZATION_LIB = True # Indicates smpl_load_model was found
except ImportError as e:
    print(f"CRITICAL Import Error: {e}")
    print("Could not import 'smpl_load_model' from SMPL.smpl_webuser.serialization.")
    print("Ensure your PYTHONPATH is configured for fake_lib, or adjust import statement.")
    HAS_SMPL_SERIALIZATION_LIB = False
    IS_CHUMPY_IMPORTED_AND_USABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_mesh(path, vertices, faces):
    """Saves a mesh to an OBJ file."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(str(path), mesh)
    print(f"Saved mesh to {path}")

def save_ply(path, vertices):
    """Saves a point cloud to a PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"Saved PLY to {path}")

def load_ply_as_torch(path, device=DEVICE):
    """Loads a PLY file as a PyTorch tensor."""
    pcd = o3d.io.read_point_cloud(str(path))
    return torch.tensor(np.asarray(pcd.points), dtype=torch.float32).to(device)

def load_obj_as_torch(path, device=DEVICE):
    """Loads an OBJ file's vertices and faces."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32).to(device)
    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int64).to(device)
    return vertices, faces
    
def _to_tensor_on_device(data, device, dtype=torch.float32):
    """Helper to convert data (numpy, chumpy.Ch, or tensor) to tensor on device."""
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype, device=device)
    elif isinstance(data, torch.Tensor):
        return data.clone().detach().to(dtype=dtype, device=device)
    elif IS_CHUMPY_IMPORTED_AND_USABLE and isinstance(data, ch.Ch): # Check if it's a chumpy object
        try:
            return torch.tensor(data.r, dtype=dtype, device=device) # .r gives numpy array
        except AttributeError:
            raise TypeError(f"Unsupported chumpy object type: {type(data)}. Expected .r attribute.")
    else:
        # Fallback for data that might be a Chumpy-like object without being ch.Ch (e.g. if chumpy import failed but fake_lib has partial chumpy)
        if hasattr(data, 'r'): # Check for .r attribute as a last resort
             try:
                 return torch.tensor(data.r, dtype=dtype, device=device)
             except Exception as e_r:
                 raise TypeError(f"Unsupported data type: {type(data)}. Attempted .r access failed: {e_r}")
        raise TypeError(f"Unsupported data type: {type(data)}. Expected numpy.ndarray, torch.Tensor, or chumpy.ch.Ch.")

def get_smal_outputs(base_model_definition_dict: dict, 
                     pkl_params_path: Path, 
                     device: torch.device, 
                     default_betas_from_family=None):
    """Loads model using smpl_load_model with params from pkl_path and base definition."""
    
    _, pkl_betas, pkl_pose = load_model_from_pkl(str(pkl_params_path))
    # pkl_trans is often not present, default to zeros if not in pkl schema for load_model_from_pkl
    # However, load_model_from_pkl doesn't return trans, so we assume it's part of pose or not used here.
    # The `smpl_load_model` from fake_lib will default trans to np.zeros(3) if not in its input dict.

    current_processing_dict = base_model_definition_dict.copy()

    current_processing_dict['pose'] = pkl_pose

    # Determine number of betas from shapedirs if present, else default to 10
    if 'shapedirs' in base_model_definition_dict and hasattr(base_model_definition_dict['shapedirs'], 'shape'):
        base_beta_shape_len = base_model_definition_dict['shapedirs'].shape[-1]
    else: # Fallback if shapedirs is not as expected or missing
        base_beta_shape_len = 10 
        print(f"Warning: 'shapedirs' not found or in unexpected format in base_model_definition_dict. Assuming {base_beta_shape_len} betas.")

    # Beta precedence: PKL file -> family_override -> base_dict default -> zeros
    if pkl_betas is not None and len(pkl_betas) == base_beta_shape_len:
        current_processing_dict['betas'] = pkl_betas
    elif default_betas_from_family is not None and len(default_betas_from_family) == base_beta_shape_len:
        current_processing_dict['betas'] = default_betas_from_family
    elif 'betas' in current_processing_dict and len(current_processing_dict['betas']) == base_beta_shape_len:
        pass # Use betas already in current_processing_dict (from base_model_definition_dict)
    else:
        print(f"Warning: Betas from PKL ({len(pkl_betas) if pkl_betas is not None else 'None'}) or family ({len(default_betas_from_family) if default_betas_from_family is not None else 'None'}) do not match expected length {base_beta_shape_len}. Using zeros.")
        current_processing_dict['betas'] = np.zeros(base_beta_shape_len)
    
    # Trans is usually set to zeros if not specified, smpl_load_model handles this default.
    # If your PKL contains 'trans', you would load it and set it here:
    # current_processing_dict['trans'] = pkl_trans_if_available

    model_obj = smpl_load_model(current_processing_dict)

    raw_verts = model_obj # The object itself is the vertices (chumpy object)
    raw_joints = model_obj.J_transformed
    faces_np = model_obj.f # This is already a numpy array from the SMPL model structure
    raw_weights = model_obj.weights
    # kintree_table is also on model_obj for skeleton parent info

    verts = _to_tensor_on_device(raw_verts, device, dtype=torch.float32)
    joints = _to_tensor_on_device(raw_joints, device, dtype=torch.float32)
    faces_torch = torch.tensor(faces_np, dtype=torch.long, device=device)
    skinning_weights = _to_tensor_on_device(raw_weights, device, dtype=torch.float32)
    
    kintree_table_np = np.asarray(model_obj.kintree_table.r if IS_CHUMPY_IMPORTED_AND_USABLE and hasattr(model_obj.kintree_table, 'r') else model_obj.kintree_table)

    return verts, faces_torch, skinning_weights, joints, kintree_table_np


def point_cloud_registration(source_points: torch.Tensor, target_points: torch.Tensor, 
                             num_iterations: int = 100, lr: float = 0.01, device: torch.device = DEVICE):
    """Aligns source_points to target_points using ICP-like optimization."""
    source_points = source_points.clone().detach().to(device)
    target_points = target_points.clone().detach().to(device)

    rot_vector = torch.zeros(3, requires_grad=True, device=device)  
    translation = torch.zeros(3, requires_grad=True, device=device) 
    optimizer = torch.optim.Adam([rot_vector, translation], lr=lr)

    print("Starting point cloud registration...")
    for i in range(num_iterations):
        optimizer.zero_grad()
        R = so3_exponential_map(rot_vector.unsqueeze(0)).squeeze(0)  
        transformed_source = (source_points @ R.T) + translation  
        
        loss, _ = chamfer_distance(transformed_source.unsqueeze(0), target_points.unsqueeze(0))
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"Iteration {i+1}/{num_iterations}: Loss = {loss.item():.6f}")

    R_final = so3_exponential_map(rot_vector.unsqueeze(0)).squeeze(0).detach()
    t_final = translation.detach()
    print("Registration finished.")
    return R_final, t_final

def compute_interpolation_coeffs(target_verts: torch.Tensor, 
                                 source_verts: torch.Tensor, 
                                 K: int = 5, 
                                 device: torch.device = DEVICE):
    """
    Computes interpolation coefficients for target_verts based on source_verts using KNN.
    Returns:
        knn_indices: (N_target, K) indices of K nearest source vertices for each target vertex.
        knn_weights: (N_target, K) normalized inverse distance weights.
    """
    target_verts = target_verts.to(device)
    source_verts = source_verts.to(device)

    if target_verts.dim() == 2:
        target_verts = target_verts.unsqueeze(0)
    if source_verts.dim() == 2:
        source_verts = source_verts.unsqueeze(0)

    dists, idx, _ = knn_points(target_verts, source_verts, K=K, return_nn=False)
    
    dists = dists.squeeze(0) 
    idx = idx.squeeze(0)   

    weights = 1.0 / (dists + 1e-8)
    weights_normalized = weights / torch.sum(weights, dim=1, keepdim=True)
    
    return idx, weights_normalized

def apply_interpolation(source_values: torch.Tensor, 
                        knn_indices: torch.Tensor, 
                        knn_weights: torch.Tensor):
    """
    Applies interpolation to source_values using precomputed KNN indices and weights.
    source_values: (N_source, D_features)
    knn_indices: (N_target, K)
    knn_weights: (N_target, K)
    Returns:
        interpolated_values: (N_target, D_features)
    """
    K = knn_indices.shape[1]
    N_target = knn_indices.shape[0]
    
    gathered_values = source_values[knn_indices.view(-1)].view(N_target, K, -1)
    
    interpolated = torch.sum(knn_weights.unsqueeze(-1) * gathered_values, dim=1)
    return interpolated


def visualize_mesh_with_weights(mesh: o3d.geometry.TriangleMesh, 
                                weights_for_one_joint: np.ndarray, 
                                title="Mesh with Skinning Weights"):
    """Visualizes a mesh, coloring vertices by given weights."""
    vis_mesh = o3d.geometry.TriangleMesh()
    vis_mesh.vertices = mesh.vertices
    vis_mesh.triangles = mesh.triangles
    vis_mesh.compute_vertex_normals()

    weights_norm = (weights_for_one_joint - weights_for_one_joint.min()) / \
                   (weights_for_one_joint.max() - weights_for_one_joint.min() + 1e-8)
    
    colors = np.zeros((len(vis_mesh.vertices), 3))
    colors[:, 0] = weights_norm 
    
    vis_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([vis_mesh], window_name=title)

def visualize_mesh_with_max_joint_influence(mesh: o3d.geometry.TriangleMesh, 
                                            full_skinning_weights: np.ndarray, 
                                            title="Mesh with Max Joint Influence"):
    """Visualizes a mesh, coloring vertices by the joint with max influence."""
    if full_skinning_weights.ndim != 2 or full_skinning_weights.shape[0] != len(mesh.vertices):
        print("Error: full_skinning_weights shape mismatch with mesh vertices.")
        print(f"Weights shape: {full_skinning_weights.shape}, Expected num_vertices: {len(mesh.vertices)}")
        o3d.visualization.draw_geometries([mesh], window_name=title + " (Error - Check Logs)")
        return

    vis_mesh = o3d.geometry.TriangleMesh()
    vis_mesh.vertices = mesh.vertices
    vis_mesh.triangles = mesh.triangles
    vis_mesh.compute_vertex_normals()

    num_vertices = full_skinning_weights.shape[0]
    num_joints = full_skinning_weights.shape[1]

    # Find the joint with maximum influence for each vertex
    max_influence_joint_indices = np.argmax(full_skinning_weights, axis=1)

    # Create a color map for joints (using a simple repeating pattern for now)
    # More sophisticated color maps can be used for better distinction if num_joints is large.
    # Example: using matplotlib's colormaps like 'tab20' or 'viridis'
    try:
        import matplotlib.cm
        # Use a perceptually uniform colormap if many joints, or a qualitative one for fewer.
        if num_joints <= 20:
            # Qualitative colormap for up to 20 distinct colors
            joint_colors = matplotlib.cm.get_cmap('tab20')(np.linspace(0, 1, num_joints))[:, :3] # Get RGB
        else:
            # Perceptually uniform colormap for more joints, cycle if needed
            joint_colors = matplotlib.cm.get_cmap('viridis')(np.linspace(0, 1, num_joints))[:, :3]
    except ImportError:
        print("Matplotlib not found, using basic cyclic colors for joint visualization.")
        # Basic cyclic colors if matplotlib is not available
        base_colors = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], 
            [1, 1, 0], [0, 1, 1], [1, 0, 1],
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]
        ])
        joint_colors = np.zeros((num_joints, 3))
        for i in range(num_joints):
            joint_colors[i] = base_colors[i % len(base_colors)]

    vertex_colors_np = joint_colors[max_influence_joint_indices]
    
    vis_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_np)
    print(f"Visualizing max joint influence. Each color represents a different joint out of {num_joints} joints.")
    o3d.visualization.draw_geometries([vis_mesh], window_name=title)

def visualize_mesh_with_skeleton(mesh: o3d.geometry.TriangleMesh, 
                                 joints: np.ndarray, 
                                 parents_kintree_row: np.ndarray, 
                                 title="Mesh with Skeleton"):
    """Visualizes a mesh with its skeleton."""
    vis_mesh = o3d.geometry.TriangleMesh()
    vis_mesh.vertices = mesh.vertices
    vis_mesh.triangles = mesh.triangles
    vis_mesh.compute_vertex_normals()
    vis_mesh.paint_uniform_color([0.8, 0.8, 0.8]) 

    lines = []
    points_np = joints # Should be numpy already for o3d
    for i, p_idx in enumerate(parents_kintree_row):
        # Ensure p_idx is a valid index before using it
        p_idx_int = int(p_idx)
        if p_idx_int != -1 and p_idx_int < len(points_np) and i < len(points_np):
            lines.append([p_idx_int, i])
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_np),
        lines=o3d.utility.Vector2iVector(lines),
    )
    colors = [[1, 0, 0] for _ in range(len(lines))] 
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    joint_spheres = []
    for joint_pos in joints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) 
        sphere.translate(joint_pos)
        sphere.paint_uniform_color([0, 0, 1]) 
        joint_spheres.append(sphere)

    o3d.visualization.draw_geometries([vis_mesh, line_set] + joint_spheres, window_name=title)

def arap_post_process(initial_verts_deformed: torch.Tensor,
                      rest_verts: torch.Tensor,
                      faces: torch.Tensor,
                      num_iters: int = 10,
                      device: torch.device = torch.device("cpu")):
    """
    Applies As-Rigid-As-Possible (ARAP) post-processing to a mesh.

    Args:
        initial_verts_deformed: Tensor of shape (N, 3), the initial distorted vertex positions.
        rest_verts: Tensor of shape (N, 3), vertex positions of the rest shape.
        faces: Tensor of shape (F, 3), mesh faces.
        num_iters: Number of ARAP iterations.
        device: PyTorch device.

    Returns:
        Tensor of shape (N, 3), the ARAP-processed vertex positions.
    """
    n_verts = rest_verts.shape[0]
    V_deformed = initial_verts_deformed.clone().to(device=device, dtype=rest_verts.dtype)
    V_rest = rest_verts.clone().to(device=device, dtype=rest_verts.dtype)
    faces_device = faces.clone().to(device=device)

    # Create PyTorch3D mesh from rest vertices to compute cotangent Laplacian
    # p3d_mesh_rest = Meshes(verts=[V_rest], faces=[faces_device]) # This line is not needed for cot_laplacian directly
    
    # L_sparse is the (N,N) cotangent Laplacian matrix: L_ii = sum_j w_ij, L_ij = -w_ij
    # It's critical that V_rest is used for cotangent weights calculation.
    # Corrected call to cot_laplacian:
    L_sparse, _ = cot_laplacian(V_rest, faces_device) # Pass verts and faces directly
    L_sparse = L_sparse.coalesce() # cot_laplacian returns (L, inv_areas)
    L_dense = L_sparse.to_dense()

    # Build adjacency list from faces for iterating neighbors
    adj_list = [[] for _ in range(n_verts)]
    for face_idx in range(faces_device.shape[0]):
        face_verts = faces_device[face_idx]
        v0, v1, v2 = face_verts[0].item(), face_verts[1].item(), face_verts[2].item()
        adj_list[v0].extend([v1, v2])
        adj_list[v1].extend([v0, v2])
        adj_list[v2].extend([v0, v1])
    
    for i in range(n_verts): # Unique, sorted neighbors
        adj_list[i] = sorted(list(set(adj_list[i])))

    rotations = torch.eye(3, device=device, dtype=V_rest.dtype).unsqueeze(0).repeat(n_verts, 1, 1)

    for iteration in range(num_iters):
        print(f"ARAP Iteration {iteration + 1}/{num_iters}")
        
        # 1. Local step: Estimate rotations R_i for each vertex
        for i in range(n_verts):
            if not adj_list[i]: # Handle isolated vertices
                rotations[i] = torch.eye(3, device=device, dtype=V_rest.dtype)
                continue

            S_i = torch.zeros((3, 3), device=device, dtype=V_rest.dtype)
            for j in adj_list[i]:
                w_ij = -L_sparse[i, j].item() # w_ij = -L_ij for i != j
                                            # Cotan weights should be non-negative.
                                            # If w_ij becomes negative due to numerical precision with L_sparse, clamp or warn.
                if w_ij < 0: # Should ideally not happen with correct cot_laplacian
                    w_ij = 0.0

                p_ij = V_rest[j] - V_rest[i]    # Edge vector in rest shape
                q_ij = V_deformed[j] - V_deformed[i]  # Edge vector in current deformed shape
                S_i += w_ij * torch.outer(p_ij, q_ij) # Covariance matrix component: w_ij * p_ij * q_ij^T
            
            try:
                U, _, Vh = torch.linalg.svd(S_i) # S_i = U Sigma Vh
                R_val = Vh.T @ U.T
                
                # Ensure proper rotation (det(R_val) = +1)
                if torch.det(R_val) < 0:
                    Vh_corrected = Vh.clone()
                    Vh_corrected[-1, :] *= -1 # Flip the last row of Vh (equivalent to flipping sign of last col of V)
                    R_val = Vh_corrected.T @ U.T
                rotations[i] = R_val
            except torch.linalg.LinAlgError: # SVD failed (e.g. S_i is zero or ill-conditioned)
                rotations[i] = torch.eye(3, device=device, dtype=V_rest.dtype) # Default to identity

        # 2. Global step: Update V_deformed by solving L_dense @ V_new = B_rhs
        B_rhs = torch.zeros((n_verts, 3), device=device, dtype=V_rest.dtype)
        for i in range(n_verts):
            if not adj_list[i]:
                # For isolated vertices, B_rhs[i] can be L_dense[i,i] * V_deformed[i]
                # to keep them fixed if L_dense[i,i] is non-zero.
                # If L_dense[i,i] is zero (truly isolated), lstsq should handle it.
                if L_dense[i,i].abs().item() > 1e-6: # Check if L_dense[i,i] is not effectively zero
                     B_rhs[i] = L_dense[i,i] * V_deformed[i]
                else: # Truly isolated or L_ii is zero, let it be zero, lstsq will find min norm solution.
                     B_rhs[i] = torch.zeros(3, device=device, dtype=V_rest.dtype)
                continue

            b_i_sum = torch.zeros(3, device=device, dtype=V_rest.dtype)
            for j in adj_list[i]:
                w_ij = -L_sparse[i, j].item()
                if w_ij < 0: w_ij = 0.0
                
                # RHS term component: 0.5 * w_ij * (R_i + R_j) @ (v_rest_i - v_rest_j)
                term_vector_rest = V_rest[i] - V_rest[j] # Edge (i,j) in rest shape as (v_i - v_j)
                transformed_term = 0.5 * w_ij * (rotations[i] + rotations[j]) @ term_vector_rest
                b_i_sum += transformed_term
            B_rhs[i] = b_i_sum
        
        try:
            # Solve L_dense @ V_new = B_rhs using least squares to handle singularity of L
            V_new = torch.linalg.lstsq(L_dense, B_rhs).solution
            V_deformed = V_new
        except torch.linalg.LinAlgError as e:
            print(f"Global step failed in ARAP iteration {iteration + 1}: {e}. Using previous V_deformed.")
            # Continue with V_deformed from the previous iteration or initial V_deformed if first iter
            pass # V_deformed remains unchanged for this iteration

    return V_deformed.cpu()

def spring_energy_post_process(initial_verts_deformed: torch.Tensor,
                               rest_verts: torch.Tensor,
                               faces_tensor: torch.Tensor,
                               num_iters: int = 50,
                               lr: float = 0.001,
                               device: torch.device = torch.device("cpu")):
    """
    Applies a spring energy model to preserve edge lengths from a rest shape.

    Args:
        initial_verts_deformed: Tensor of shape (N, 3), initial distorted vertex positions.
        rest_verts: Tensor of shape (N, 3), vertex positions of the rest shape.
        faces_tensor: Tensor of shape (F, 3), mesh faces.
        num_iters: Number of optimization iterations.
        lr: Learning rate for the optimizer.
        device: PyTorch device.

    Returns:
        Tensor of shape (N, 3), the processed vertex positions.
    """
    V_processed = initial_verts_deformed.clone().to(device=device, dtype=rest_verts.dtype).requires_grad_(True)
    V_rest_device = rest_verts.clone().to(device=device, dtype=rest_verts.dtype)
    faces_device = faces_tensor.clone().to(device=device)

    # Extract unique edges
    edge_list = []
    for face_idx in range(faces_device.shape[0]):
        f = faces_device[face_idx]
        edge_list.append(tuple(sorted((f[0].item(), f[1].item()))))
        edge_list.append(tuple(sorted((f[1].item(), f[2].item()))))
        edge_list.append(tuple(sorted((f[2].item(), f[0].item()))))
    
    unique_edges_tuples = sorted(list(set(edge_list)))
    if not unique_edges_tuples:
        print("Warning: No unique edges found. Returning initial vertices.")
        return initial_verts_deformed.cpu()
        
    unique_edges = torch.tensor(unique_edges_tuples, dtype=torch.long, device=device)

    # Calculate rest edge lengths
    rest_edge_vectors = V_rest_device[unique_edges[:, 0]] - V_rest_device[unique_edges[:, 1]]
    rest_edge_lengths = torch.norm(rest_edge_vectors, p=2, dim=1)

    optimizer = torch.optim.Adam([V_processed], lr=lr)

    print("Starting spring energy optimization...")
    for iteration in range(num_iters):
        optimizer.zero_grad()
        
        current_edge_vectors = V_processed[unique_edges[:, 0]] - V_processed[unique_edges[:, 1]]
        current_edge_lengths = torch.norm(current_edge_vectors, p=2, dim=1)
        
        loss = torch.sum((current_edge_lengths - rest_edge_lengths)**2)
        
        loss.backward()
        optimizer.step()
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Spring Iteration {iteration + 1}/{num_iters}: Loss = {loss.item():.6f}")

    print("Spring energy optimization finished.")
    return V_processed.detach().cpu()

def laplacian_regularization_post_process(initial_verts_deformed: torch.Tensor,
                                          rest_verts_for_laplacian_calc: torch.Tensor,
                                          faces_tensor: torch.Tensor,
                                          smoothness_param_w: float = 1.0,
                                          device: torch.device = torch.device("cpu")):
    """
    Applies Laplacian regularization to smooth a mesh while preserving shape.
    Solves (w * L_transpose @ L + I) @ V_out = V_in
    where L is cot_laplacian computed on rest_verts_for_laplacian_calc.

    Args:
        initial_verts_deformed (V_in): Tensor (N,3), the mesh to be smoothed.
        rest_verts_for_laplacian_calc: Tensor (N,3), vertices of the mesh whose geometry
                                       defines the Laplacian operator (e.g., aligned scan).
        faces_tensor: Tensor (F,3), mesh faces.
        smoothness_param_w (w): Weight for the smoothness term (L^T L).
        device: PyTorch device.

    Returns:
        Tensor (N,3), the smoothed vertex positions (V_out).
    """
    V_in = initial_verts_deformed.clone().to(device=device, dtype=torch.float32)
    V_lap_calc = rest_verts_for_laplacian_calc.clone().to(device=device, dtype=torch.float32)
    faces_device = faces_tensor.clone().to(device=device)
    
    n_verts = V_in.shape[0]

    print("Starting Laplacian Regularization...")
    print(f"Using smoothness_param_w: {smoothness_param_w}")

    # L is the cotangent Laplacian matrix (N, N)
    # Computed using the geometry of rest_verts_for_laplacian_calc
    L_cot, _ = cot_laplacian(V_lap_calc, faces_device) # Returns sparse tensor
    
    # System matrix A = (w * L_cot^T @ L_cot + I)
    # L_cot.T @ L_cot can be dense if L_cot is not perfectly structured or dense itself.
    # For robustness with lstsq, we might need to densify L_cot first for the transpose product.
    # However, PyTorch sparse @ sparse is efficient. Let's try with sparse ops first.
    
    # Ensure L_cot is coalesced for efficient sparse matrix multiplication
    if not L_cot.is_coalesced():
        L_cot = L_cot.coalesce()

    # L_transpose_L = L_cot.T @ L_cot # This should also be sparse
    # An alternative if L_cot.T @ L_cot is problematic or too slow to form explicitly:
    # Iterative solvers like Conjugate Gradient are good for (A.T A + lambda I)x = A.T b form.
    # Here, our system is (w * L.T L + I) V = V_in. This is a standard system form.

    # Forming L_transpose_L explicitly for now.
    # If L_cot is very large, this matrix can be large. Consider alternatives if performance is an issue.
    L_transpose_L = torch.sparse.mm(L_cot.transpose(0, 1), L_cot) # L.T @ L

    # Identity matrix (sparse or dense depending on what lstsq handles best with A)
    # torch.linalg.lstsq can handle sparse A if A.is_sparse is true, but it may densify internally.
    # Let's try constructing A as sparse first.
    identity_matrix_sparse = torch.sparse_coo_tensor(
        torch.arange(n_verts, device=device).unsqueeze(0).repeat(2,1),
        torch.ones(n_verts, device=device, dtype=torch.float32),
        (n_verts, n_verts)
    ).coalesce()

    A_matrix_sparse = smoothness_param_w * L_transpose_L + identity_matrix_sparse
    A_matrix_sparse = A_matrix_sparse.coalesce()

    # B matrix is V_in
    B_matrix = V_in

    print(f"Solving linear system for {n_verts} vertices...")
    try:
        # torch.linalg.lstsq should handle sparse A. If not, we might need to densify A.
        # Or use torch.sparse.linalg.solve if the matrix A is well-behaved (e.g. SPD for Cholesky based solvers)
        # lstsq is more general.
        V_out = torch.linalg.lstsq(A_matrix_sparse, B_matrix).solution
        
        # Check for NaNs or Infs, which can happen if the system is ill-conditioned
        if torch.isnan(V_out).any() or torch.isinf(V_out).any():
            print("Warning: NaNs or Infs detected in V_out. System might be ill-conditioned.")
            print("Falling back to dense solver for potentially better stability/error reporting.")
            A_dense = A_matrix_sparse.to_dense()
            V_out = torch.linalg.lstsq(A_dense, B_matrix).solution
            if torch.isnan(V_out).any() or torch.isinf(V_out).any():
                 print("CRITICAL: NaNs/Infs persist even with dense solver. Problem with system matrix or inputs.")
                 print("Returning V_in as fallback.")
                 return V_in.cpu().detach()

    except torch.linalg.LinAlgError as e:
        print(f"torch.linalg.LinAlgError during sparse solve: {e}")
        print("Attempting to solve with dense matrix A instead.")
        try:
            A_dense = A_matrix_sparse.to_dense()
            V_out = torch.linalg.lstsq(A_dense, B_matrix).solution
        except torch.linalg.LinAlgError as e_dense:
            print(f"CRITICAL: torch.linalg.LinAlgError even with dense solve: {e_dense}. Returning V_in.")
            return V_in.cpu().detach()
    except Exception as e_gen:
        print(f"CRITICAL: An unexpected error occurred during solve: {e_gen}. Returning V_in.")
        return V_in.cpu().detach()

    print("Laplacian Regularization finished.")
    return V_out.cpu().detach()

def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_SMPL_SERIALIZATION_LIB:
        print("CRITICAL ERROR: SMPL (or fake_lib) package providing smpl_load_model is required.")
        return

    # 1. Load base SMAL/SMPL model definition from the main model PKL
    print(f"Loading base model definition from: {args.smal_model_path}")
    try:
        with open(args.smal_model_path, 'rb') as f:
            base_model_definition_dict = pickle.load(f, encoding='latin1')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Base model PKL not found at {args.smal_model_path}")
        return
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load base model PKL from {args.smal_model_path}. Error: {e}")
        return
    
    # Extract family-specific betas if smal_data_path is provided
    family_betas_override = None
    if args.smal_data_path and args.smal_data_path.exists():
        print(f"Loading SMAL family data from: {args.smal_data_path}")
        with open(args.smal_data_path, 'rb') as f:
            smal_family_data = pickle.load(f, encoding='latin1')
        if 'cluster_means' in smal_family_data and len(smal_family_data['cluster_means']) > args.shape_family_id:
            family_betas_override = smal_family_data['cluster_means'][args.shape_family_id]
            print(f"Loaded family-specific betas for family ID {args.shape_family_id}")
        else:
            print(f"Warning: Cluster means for family ID {args.shape_family_id} not found in {args.smal_data_path}.")
    else:
        print("No SMAL family data path provided or file not found. Will use betas from PKL or defaults.")

    # 2. Get template model outputs (posed mesh, pcd, weights, joints, kintree)
    print(f"Processing template PKL: {args.template_smal_pkl}")
    template_verts, template_faces, template_skinning_weights, template_joints, template_kintree_table_np = get_smal_outputs(base_model_definition_dict, args.template_smal_pkl, DEVICE, family_betas_override)

    template_smal_obj_path = args.template_obj_path if args.template_obj_path else args.output_dir / "template_smal.obj"
    if not args.template_obj_path or not template_smal_obj_path.exists():
        save_mesh(template_smal_obj_path, template_verts.cpu().numpy(), template_faces.cpu().numpy())
        print(f"Saved/Generated template_smal.obj at {template_smal_obj_path}")
    
    template_ply_path_actual = args.template_ply_path
    if not template_ply_path_actual or not Path(template_ply_path_actual).exists(): # Check if path string exists
        template_ply_path_actual = args.output_dir / "template_smal.ply"
        save_ply(template_ply_path_actual, template_verts.cpu().numpy())
        print(f"Saved/Generated template_smal.ply at {template_ply_path_actual}")
    
    template_smal_pcd = load_ply_as_torch(template_ply_path_actual, DEVICE)


    # 3. Load render mesh and point cloud
    render_obj_verts, render_obj_faces = load_obj_as_torch(args.render_obj_path, DEVICE)
    render_pcd = load_ply_as_torch(args.render_ply_path, DEVICE)

    # 4. Point Cloud Registration (align render_pcd to template_smal_pcd)
    R, t = point_cloud_registration(render_pcd, template_smal_pcd, 
                                    num_iterations=args.reg_iters, lr=args.reg_lr, device=DEVICE)

    # 5. Align full render_obj mesh
    aligned_render_obj_verts = (render_obj_verts @ R.T) + t
    save_mesh(args.output_dir / "aligned_render.obj", 
              aligned_render_obj_verts.cpu().numpy(), 
              render_obj_faces.cpu().numpy())

    # 6. Compute Interpolation Coefficients (from template_smal_verts to aligned_render_obj_verts)
    interp_indices, interp_coeffs = compute_interpolation_coeffs(
        aligned_render_obj_verts, 
        template_verts, 
        K=args.knn_k, 
        device=DEVICE
    )

    # 7. Transfer Skinning Weights to aligned_render_mesh
    aligned_render_skinning_weights = apply_interpolation(
        template_skinning_weights, 
        interp_indices, 
        interp_coeffs
    )
    np.save(args.output_dir / "aligned_render_weights.npy", aligned_render_skinning_weights.cpu().numpy())


    # 8. Visualization of aligned render mesh with weights and skeleton
    print("Visualizing aligned render mesh with skinning weights and skeleton...")
    
    # Create Open3D mesh for visualization directly from the aligned vertices and original faces
    # This ensures consistency with the vertices used for weight calculation.
    aligned_o3d_mesh_for_vis = o3d.geometry.TriangleMesh()
    aligned_o3d_mesh_for_vis.vertices = o3d.utility.Vector3dVector(aligned_render_obj_verts.cpu().numpy())
    aligned_o3d_mesh_for_vis.triangles = o3d.utility.Vector3iVector(render_obj_faces.cpu().numpy())
    aligned_o3d_mesh_for_vis.compute_vertex_normals() # Important for visualization
    
    # Visualize weights for a specific joint (e.g., joint 5 instead of 0)
    joint_to_visualize_idx = 5 
    num_joints_in_weights = aligned_render_skinning_weights.shape[1]
    if joint_to_visualize_idx >= num_joints_in_weights:
        print(f"Warning: joint_to_visualize_idx {joint_to_visualize_idx} is out of bounds for weights with {num_joints_in_weights} joints. Visualizing joint 0 instead.")
        joint_to_visualize_idx = 0

    print(f"Visualizing skinning weights for joint index: {joint_to_visualize_idx}")
    visualize_mesh_with_weights(
        aligned_o3d_mesh_for_vis,
        aligned_render_skinning_weights[:, joint_to_visualize_idx].cpu().numpy(),
        title=f"Aligned Render Mesh - Skinning Weights (Joint {joint_to_visualize_idx})"
    )

    # Visualize max joint influence on the aligned mesh
    print("Visualizing max joint influence on aligned render mesh...")
    visualize_mesh_with_max_joint_influence(
        aligned_o3d_mesh_for_vis,
        aligned_render_skinning_weights.cpu().numpy(), # Pass the full weight matrix
        title="Aligned Render Mesh - Max Joint Influence"
    )
    
    # Use the first row of kintree_table for parent indices
    # Ensure template_kintree is a numpy array before indexing
    parents_info_np = template_kintree_table_np[0, :] 

    visualize_mesh_with_skeleton(
        aligned_o3d_mesh_for_vis, # Use the directly created mesh here as well
        template_joints.cpu().numpy(),
        parents_info_np, 
        title="Aligned Render Mesh - Template Skeleton (Posed)"
    )

    # 9. Retargeting to new pose/shape from --target_smal_pkl
    if args.target_smal_pkl:
        print(f"Retargeting using {args.target_smal_pkl}...")
        target_verts, _, _, target_joints, target_kintree_table_np = get_smal_outputs(base_model_definition_dict, args.target_smal_pkl, DEVICE, family_betas_override)
        
        retargeted_render_verts = apply_interpolation(
            target_verts, 
            interp_indices,
            interp_coeffs
        )
        save_mesh(args.output_dir / "retargeted_render_initial.obj", 
                  retargeted_render_verts.cpu().numpy(), 
                  render_obj_faces.cpu().numpy())
        print(f"Saved initial retargeted_render_initial.obj based on {args.target_smal_pkl}")

        final_retargeted_verts_to_visualize = retargeted_render_verts # Default to initial
        vis_title_suffix = f"Initial ({args.target_smal_pkl.name})"
        output_obj_filename = "retargeted_render_initial.obj" # Already saved

        if args.laplacian_reg_weight >= 0: # Use laplacian if weight is non-negative (0 means only data term)
            print("Applying Laplacian Regularization post-processing...")
            laplacian_reg_verts = laplacian_regularization_post_process(
                initial_verts_deformed=retargeted_render_verts.to(DEVICE),
                rest_verts_for_laplacian_calc=aligned_render_obj_verts.to(DEVICE),
                faces_tensor=render_obj_faces.to(DEVICE),
                smoothness_param_w=args.laplacian_reg_weight,
                device=DEVICE
            )
            final_retargeted_verts_to_visualize = laplacian_reg_verts
            output_obj_filename = "retargeted_render_laplacian_reg.obj"
            save_mesh(args.output_dir / output_obj_filename,
                      laplacian_reg_verts.cpu().numpy(),
                      render_obj_faces.cpu().numpy())
            print(f"Saved Laplacian Regularized {output_obj_filename}")
            vis_title_suffix = f"Laplacian Reg. (w={args.laplacian_reg_weight}) ({args.target_smal_pkl.name})"
        
        elif args.spring_iters > 0:
            print("Applying Spring Energy post-processing to retargeted mesh...")
            spring_processed_verts = spring_energy_post_process(
                initial_verts_deformed=retargeted_render_verts.to(DEVICE),
                rest_verts=aligned_render_obj_verts.to(DEVICE),
                faces_tensor=render_obj_faces.to(DEVICE),
                num_iters=args.spring_iters,
                lr=args.spring_lr,
                device=DEVICE
            )
            final_retargeted_verts_to_visualize = spring_processed_verts
            output_obj_filename = "retargeted_render_spring.obj"
            save_mesh(args.output_dir / output_obj_filename,
                      spring_processed_verts.cpu().numpy(),
                      render_obj_faces.cpu().numpy())
            print(f"Saved Spring processed {output_obj_filename}")
            vis_title_suffix = f"Spring Processed ({args.target_smal_pkl.name})"

        elif args.arap_iters > 0: # ARAP is fallback if spring_iters is 0
            print("Applying ARAP post-processing to retargeted mesh...")
            # V_rest for ARAP is the aligned_render_obj_verts (shape before this specific target's deformation)
            # V_skin for ARAP is retargeted_render_verts
            
            arap_processed_verts = arap_post_process(
                initial_verts_deformed=retargeted_render_verts.to(DEVICE),
                rest_verts=aligned_render_obj_verts.to(DEVICE), # This is key: rest shape is the aligned render mesh
                faces=render_obj_faces.to(DEVICE),
                num_iters=args.arap_iters,
                device=DEVICE
            )
            
            final_retargeted_verts_to_visualize = arap_processed_verts # Use ARAP result
            output_obj_filename = "retargeted_render_arap.obj"
            save_mesh(args.output_dir / output_obj_filename, 
                      arap_processed_verts.cpu().numpy(), 
                      render_obj_faces.cpu().numpy())
            print(f"Saved ARAP processed {output_obj_filename}")
            vis_title_suffix = f"ARAP Processed ({args.target_smal_pkl.name})"
        # else: visual will use initial retargeted_render_verts saved as retargeted_render_initial.obj

        # Optional: Visualize final retargeted mesh (either ARAP processed or initial)
        retargeted_o3d_mesh_for_vis = o3d.geometry.TriangleMesh()
        retargeted_o3d_mesh_for_vis.vertices = o3d.utility.Vector3dVector(final_retargeted_verts_to_visualize.cpu().numpy())
        retargeted_o3d_mesh_for_vis.triangles = o3d.utility.Vector3iVector(render_obj_faces.cpu().numpy())
        retargeted_o3d_mesh_for_vis.compute_vertex_normals()

        target_parents_np = target_kintree_table_np[0, :]
        visualize_mesh_with_skeleton(
            retargeted_o3d_mesh_for_vis,
            target_joints.cpu().numpy(),
            target_parents_np,
            title=f"Retargeted Render Mesh {vis_title_suffix}"
        )
    else:
        print("No --target_smal_pkl provided, skipping final retargeting step.")

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh retargeting from SMAL/SMPL to a high-quality render mesh using smpl_webuser.serialization.")
    
    # SMAL model paths
    parser.add_argument("--smal_model_path", type=Path, default=Path("../../data/smal_CVPR2017.pkl"), help="Path to the base SMAL/SMPL .pkl model definition file (e.g., smal_CVPR2017.pkl).")
    parser.add_argument("--smal_data_path", type=Path, default=Path("../../data/smal_CVPR2017_data.pkl"), help="Path to smal_CVPR2017_data.pkl (for family-specific shapes).")
    parser.add_argument("--shape_family_id", type=int, default=3, help="SMAL shape family ID (e.g., 3 for cows, used with smal_data_path).")

    # Template SMAL inputs
    parser.add_argument("--template_smal_pkl", type=Path, default=Path("./template.pkl"), help="Path to template.pkl (pose/beta parameters for the template shape/pose).")
    parser.add_argument("--template_ply_path", type=Path, default=None, help="Path to template.ply (point cloud of template, generated if not provided).")
    parser.add_argument("--template_obj_path", type=Path, default=None, help="Path to template.obj (mesh of template, generated if not provided).")
    
    # Render mesh inputs
    parser.add_argument("--render_ply_path", type=Path, default=Path("render.ply"), help="Path to render.ply (high-res point cloud for registration source).")
    parser.add_argument("--render_obj_path", type=Path, default=Path("render.obj"), help="Path to render.obj (high-res mesh to be deformed).")

    # Target SMAL input for retargeting
    parser.add_argument("--target_smal_pkl", "--input", type=Path, default=None, help="Path to target.pkl (new pose/beta parameters for retargeting).")

    # Output directory - made required as it's crucial and user-specific for runs
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save results.")

    # Algorithm parameters
    parser.add_argument("--reg_iters", type=int, default=200, help="Number of iterations for point cloud registration.")
    parser.add_argument("--reg_lr", type=float, default=0.01, help="Learning rate for point cloud registration.")
    parser.add_argument("--knn_k", type=int, default=5, help="Number of K for KNN interpolation.")
    parser.add_argument("--arap_iters", type=int, default=0, help="Number of iterations for ARAP post-processing. Set to 0 to disable.")
    parser.add_argument("--spring_iters", type=int, default=0, help="Number of iterations for Spring Energy post-processing. Set to 0 to disable. Overrides ARAP if > 0.")
    parser.add_argument("--spring_lr", type=float, default=0.001, help="Learning rate for Spring Energy optimization.")
    parser.add_argument("--laplacian_reg_weight", type=float, default=-1.0, help="Weight for Laplacian regularization (w in (w * L^T L + I)V = V_in). Set >= 0 to enable. Overrides Spring and ARAP.")

    parsed_args = parser.parse_args()
    
    # Resolve default paths relative to the script location if they are not absolute
    script_dir = Path(__file__).parent.resolve()
    if not parsed_args.smal_model_path.is_absolute():
        parsed_args.smal_model_path = (script_dir / parsed_args.smal_model_path).resolve()
    if parsed_args.smal_data_path and not parsed_args.smal_data_path.is_absolute():
        parsed_args.smal_data_path = (script_dir / parsed_args.smal_data_path).resolve()
    if not parsed_args.template_smal_pkl.is_absolute():
        parsed_args.template_smal_pkl = (script_dir / parsed_args.template_smal_pkl).resolve()
    if parsed_args.template_ply_path and not parsed_args.template_ply_path.is_absolute():
        parsed_args.template_ply_path = (script_dir / parsed_args.template_ply_path).resolve()
    if parsed_args.template_obj_path and not parsed_args.template_obj_path.is_absolute():
        parsed_args.template_obj_path = (script_dir / parsed_args.template_obj_path).resolve()
    if not parsed_args.render_ply_path.is_absolute():
        parsed_args.render_ply_path = (script_dir / parsed_args.render_ply_path).resolve()
    if not parsed_args.render_obj_path.is_absolute():
        parsed_args.render_obj_path = (script_dir / parsed_args.render_obj_path).resolve()
    if parsed_args.target_smal_pkl and not parsed_args.target_smal_pkl.is_absolute():
         parsed_args.target_smal_pkl = (script_dir / parsed_args.target_smal_pkl).resolve()
    
    main(parsed_args) 