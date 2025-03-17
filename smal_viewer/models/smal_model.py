import numpy as np
import pickle
import scipy.sparse as sp
from pathlib import Path
import os

# Fix for chumpy with newer NumPy versions
import sys
import warnings

# Suppress FutureWarnings from NumPy
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to patch chumpy before importing it
try:
    # Add a patched version of numpy for chumpy
    class PatchedNumpy:
        def __init__(self, numpy_module):
            self.__dict__ = numpy_module.__dict__.copy()
            # Add deprecated types that chumpy expects
            self.bool = np.bool_
            self.int = np.int_
            self.float = np.float_
            self.complex = np.complex_
            self.object = np.object_
            self.unicode = np.str_
            self.str = np.str_

    # Create patched numpy
    patched_numpy = PatchedNumpy(np)
    
    # Temporarily replace numpy in sys.modules
    original_numpy = sys.modules['numpy']
    sys.modules['numpy'] = patched_numpy
    
    # Now import chumpy with the patched numpy
    import chumpy
    HAS_CHUMPY = True
    
    # Restore original numpy
    sys.modules['numpy'] = original_numpy
    
except ImportError:
    HAS_CHUMPY = False
    print("Warning: chumpy not found. Some functionality may be limited.")

class SMALModel:
    """
    SMAL (Skinned Multi-Animal Linear) model implementation for Python 3.10
    """
    
    def __init__(self, model_path, shape_family_id=1):
        """
        Initialize the SMAL model
        
        Args:
            model_path: Path to the SMAL model pickle file
            shape_family_id: Animal family ID (default: 1 for dogs)
                          0-felidae(cats); 1-canidae(dogs); 2-equidae(horses);
                          3-bovidae(cows); 4-hippopotamidae(hippos)
        """
        # Load the model
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        
        # Extract model parameters
        self.v_template = params['v_template']
        self.f = params['f']
        self.shapedirs = params['shapedirs']
        self.J_regressor = params['J_regressor']
        self.weights = params['weights']
        self.kintree_table = params['kintree_table']
        self.posedirs = params['posedirs']
        self.shape_family_id = shape_family_id
        
        # Convert sparse matrices if needed
        if sp.issparse(self.J_regressor):
            self.J_regressor = self.J_regressor.toarray()
        
        # Initialize pose and shape parameters
        self.pose = np.zeros(self.posedirs.shape[2])
        self.betas = np.zeros(self.shapedirs.shape[2])
        
        # Set shape family
        if 'shape_family_basis' in params:
            self.shape_family_basis = params['shape_family_basis']
            self.set_shape_family(shape_family_id)
        
        # Calculate joint positions
        self.J = np.matmul(self.J_regressor, self.v_template)
        
        # Initialize transformations
        self.num_joints = self.J.shape[0]
        
    def set_shape_family(self, family_id):
        """Set the shape family for the model"""
        if hasattr(self, 'shape_family_basis'):
            # Reset shape parameters
            self.betas = np.zeros_like(self.betas)
            
            # Apply custom shape parameters based on family ID
            # According to the demo.py:
            # 0-felidae(cats); 1-canidae(dogs); 2-equidae(horses);
            # 3-bovidae(cows); 4-hippopotamidae(hippos)
            if family_id == 0:  # Cat (was 2 in your code)
                # Make the cat more cat-like
                self.betas[0] = -0.5  # Overall size (smaller)
                self.betas[1] = -0.5  # Leg length (shorter)
                self.betas[2] = -0.5  # Body width (thinner)
                self.betas[3] = 0.5   # Head size (larger)
                self.betas[4] = 1.5   # Tail length (longer)
            elif family_id == 1:  # Dog (was 1 in your code)
                # Make the dog more dog-like
                self.betas[0] = 0.0   # Overall size
                self.betas[1] = -1.0  # Leg length (shorter)
                self.betas[2] = 0.5   # Body width
                self.betas[3] = 0.0   # Head size
                self.betas[4] = 1.0   # Tail length
            elif family_id == 2:  # Horse (was 3 in your code)
                # Make the horse more horse-like
                self.betas[0] = 1.0   # Overall size (larger)
                self.betas[1] = 2.0   # Leg length (longer)
                self.betas[2] = 0.5   # Body width
                self.betas[3] = 0.5   # Head size
                self.betas[4] = 0.5   # Tail length
            elif family_id == 3:  # Cow (was 4 in your code)
                # Make the cow more cow-like
                self.betas[0] = 1.5   # Overall size (larger)
                self.betas[1] = 0.5   # Leg length
                self.betas[2] = 1.5   # Body width (wider)
                self.betas[3] = 0.0   # Head size
                self.betas[4] = 0.5   # Tail length
            elif family_id == 4:  # Hippo (was 5 in your code)
                # Make the hippo more hippo-like
                self.betas[0] = 2.0   # Overall size (much larger)
                self.betas[1] = -1.0  # Leg length (shorter)
                self.betas[2] = 2.0   # Body width (much wider)
                self.betas[3] = 0.5   # Head size
                self.betas[4] = -0.5  # Tail length (shorter)
            
            print(f"Applied custom shape parameters for family {family_id}: {self.betas[:5]}")
    
    def update_shape(self, betas=None):
        """Update the shape of the model based on shape parameters"""
        if betas is not None:
            self.betas = betas
        
        # Apply shape blend shapes
        v_shaped = self.v_template + np.matmul(self.shapedirs, self.betas)
        
        # Update joint positions
        self.J = np.matmul(self.J_regressor, v_shaped)
        
        return v_shaped
    
    def rodrigues(self, r):
        """
        Rodrigues formula for converting axis-angle to rotation matrix
        
        Args:
            r: Axis-angle representation (3D vector)
        
        Returns:
            3x3 rotation matrix
        """
        # Ensure r is a numpy array with the right shape
        r = np.asarray(r, dtype=np.float64).flatten()
        
        if r.size != 3:
            raise ValueError(f"Input must be a 3D vector, got shape {r.shape}")
        
        theta = np.linalg.norm(r)
        
        if theta < 1e-8:
            # If rotation is very small, return identity
            return np.eye(3)
        
        # Normalize rotation axis
        k = r / theta
        
        # Cross product matrix
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        
        return R
    
    def update_pose(self, pose=None):
        """Update the pose of the model based on pose parameters"""
        if pose is not None:
            self.pose = pose
        
        # Get shaped vertices
        v_shaped = self.update_shape()
        
        # Initialize posed vertices
        v_posed = np.copy(v_shaped)
        
        # Apply pose blend shapes
        R = np.zeros((self.num_joints, 3, 3))
        for i in range(self.num_joints):
            joint_pose = self.pose[i*3:(i+1)*3]
            R[i] = self.rodrigues(joint_pose)
        
        # Apply pose blend shapes
        pose_feature = (R[1:] - np.eye(3)).reshape(-1)
        v_posed += np.matmul(self.posedirs, pose_feature)
        
        # Apply linear blend skinning
        T = np.zeros((self.num_joints, 4, 4))
        
        # Root joint transformation
        T[0, :3, :3] = R[0]
        T[0, :3, 3] = self.J[0]
        T[0, 3, 3] = 1.0
        
        # Other joints
        for i in range(1, self.num_joints):
            parent = self.kintree_table[0, i]
            T[i, :3, :3] = R[i]
            T[i, :3, 3] = self.J[i] - np.matmul(R[i], self.J[parent])
            T[i, 3, 3] = 1.0
            
            # Global transformation
            T[i] = np.matmul(T[parent], T[i])
        
        # Apply skinning
        rest_shape_h = np.hstack((v_posed, np.ones((v_posed.shape[0], 1))))
        
        # Initialize transformed vertices
        v_transformed = np.zeros_like(rest_shape_h)
        
        # Apply skinning transformation
        for i in range(self.num_joints):
            v_transformed += self.weights[:, i:i+1] * np.matmul(rest_shape_h, T[i].T)
        
        return v_transformed[:, :3]
    
    def set_params(self, pose=None, betas=None):
        """Set pose and shape parameters and update the model"""
        if pose is not None:
            self.pose = pose
        if betas is not None:
            self.betas = betas
        
        return self.update_pose()
    
    def get_transformed_joints(self):
        """
        Get joint positions transformed by the current pose and shape
        
        Returns:
            numpy.ndarray: Transformed joint positions
        """
        # This method needs to exactly match the transformation pipeline in update_pose()
        
        # First update joint positions based on shape
        v_shaped = self.v_template + np.matmul(self.shapedirs, self.betas)
        J = np.matmul(self.J_regressor, v_shaped)
        
        # Initialize transformations
        T = np.zeros((self.num_joints, 4, 4))
        
        # Calculate rotation matrices
        R = np.zeros((self.num_joints, 3, 3))
        for i in range(self.num_joints):
            joint_pose = self.pose[i*3:(i+1)*3]
            R[i] = self.rodrigues(joint_pose)
        
        # Root joint transformation
        T[0, :3, :3] = R[0]
        T[0, :3, 3] = J[0]
        T[0, 3, 3] = 1.0
        
        # Other joints
        for i in range(1, self.num_joints):
            parent = self.kintree_table[0, i]
            T[i, :3, :3] = R[i]
            T[i, :3, 3] = J[i] - np.matmul(R[i], J[parent])
            T[i, 3, 3] = 1.0
            
            # Global transformation
            T[i] = np.matmul(T[parent], T[i])
        
        # Extract joint positions from transformation matrices
        transformed_joints = T[:, :3, 3]
        
        # Apply the same rotation as in paintGL to match the model orientation
        # Rotate -90 degrees around X axis
        rotation = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        
        # Apply rotation to each joint
        rotated_joints = np.zeros_like(transformed_joints)
        for i in range(len(transformed_joints)):
            rotated_joints[i] = np.matmul(rotation, transformed_joints[i])
        
        return rotated_joints

    def forward(self, pose, betas=None, trans=None, del_v=None, keypoints=None):
        """
        Forward pass for SMAL model

        Args:
            pose: pose parameters - rotation angles for joints
            betas: shape parameters
            trans: global translation
            del_v: vertex offsets
            keypoints: keypoints for keypoint loss
        """
        # Update shape parameters if provided
        if betas is not None:
            # Simple dimension check
            expected_dim = self.shapedirs.shape[2]
            if betas.shape[0] != expected_dim:
                print(f"WARNING: Shape parameter dimension mismatch. Got {betas.shape[0]}, expected {expected_dim}.")
                # Simple padding/truncation to match dimensions
                padded_betas = np.zeros(expected_dim, dtype=np.float64)
                min_dim = min(betas.shape[0], expected_dim)
                padded_betas[:min_dim] = betas[:min_dim]
                betas = padded_betas
            
            # Update the model's shape parameters
            self.betas = betas
            
            # Apply shape blend shapes
            v_shaped = self.v_template + np.matmul(self.shapedirs, betas)
        else:
            # Use current betas if none provided
            v_shaped = self.v_template + np.matmul(self.shapedirs, self.betas)
        
        # Update joint positions based on new shape
        self.J = np.matmul(self.J_regressor, v_shaped)
        
        # If pose is provided, update the pose as well
        if pose is not None:
            self.pose = pose
        
        # Apply the pose to get the final vertices
        return self.update_pose()

    def initialize_for_cow(self):
        """Initialize the model specifically for cow visualization using SMAL data"""
        # Try to load the SMAL data file to get actual cow parameters
        from pathlib import Path
        import pickle
        
        # Possible locations for the data file
        possible_data_paths = [
            Path("./data/smal_CVPR2017_data.pkl"),
            Path("./smal_CVPR2017_data.pkl"),
            Path("../data/smal_CVPR2017_data.pkl"),
        ]
        
        # Try to find and load the data file
        for data_path in possible_data_paths:
            if data_path.exists():
                try:
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')
                    
                    # Bovidae (cows) are family 3
                    if 'cluster_means' in data and len(data['cluster_means']) > 3:
                        cow_betas = data['cluster_means'][3]
                        print(f"Using actual bovidae family shape parameters from SMAL data")
                        # Update the model with the cow shape parameters
                        self.betas = cow_betas
                        return self.forward(pose=self.pose, betas=cow_betas)
                except Exception as e:
                    print(f"Error loading SMAL data: {e}")
        
        # Fallback to the set_shape_family method if data file not found
        print("SMAL data file not found, using predefined cow parameters")
        self.set_shape_family(3)  # 3 is the bovidae/cow family ID
        return self.forward(pose=self.pose, betas=self.betas)

    def visualize_shape_parameter_effects(self, parameter_index, values=[-2.0, -1.0, 0.0, 1.0, 2.0]):
        """
        Visualize the effect of a specific shape parameter
        
        Args:
            parameter_index: Index of the shape parameter to visualize
            values: List of values to try for the parameter
            
        Returns:
            List of vertices for each value
        """
        results = []
        
        # Store original betas
        original_betas = self.betas.copy()
        
        for value in values:
            # Create a copy of the original betas
            test_betas = original_betas.copy()
            
            # Set the specified parameter to the test value
            test_betas[parameter_index] = value
            
            # Apply the shape parameters
            vertices = self.forward(pose=self.pose, betas=test_betas)
            
            # Store the result
            results.append((value, vertices))
            
            print(f"Parameter {parameter_index} = {value}: Shape effect applied")
        
        # Restore original betas
        self.forward(pose=self.pose, betas=original_betas)
        
        return results 