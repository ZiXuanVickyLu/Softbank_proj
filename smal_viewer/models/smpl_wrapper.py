import numpy as np

class SMPLWrapper:
    """
    Wrapper for SMPL model to make it compatible with our viewer
    """
    
    def __init__(self, smpl_model):
        """
        Initialize the wrapper with a SMPL model
        
        Args:
            smpl_model: SMPL model from SMPL.smpl_webuser.serialization.load_model
        """
        self.smpl_model = smpl_model
        self.f = smpl_model.f
        
        # Store the current pose and shape parameters
        # Handle both Chumpy objects and NumPy arrays
        if hasattr(smpl_model.pose, 'r'):
            # Chumpy object case
            self.pose = np.array(smpl_model.pose.r)
            self.betas = np.array(smpl_model.betas.r)
        else:
            # NumPy array case
            self.pose = np.array(smpl_model.pose)
            self.betas = np.array(smpl_model.betas)
        
        # Copy the kinematic tree table if available
        if hasattr(smpl_model, 'kintree_table'):
            self.kintree_table = np.array(smpl_model.kintree_table)
        else:
            # Create a default kinematic tree if not available
            # This is a simplified version - adjust based on your model
            self.kintree_table = np.zeros((2, 24), dtype=np.int32)  # Assuming 24 joints
            # Set parent-child relationships (this is a simplified example)
            self.kintree_table[0, 1:] = np.arange(23)  # Each joint's index
            self.kintree_table[1, 1:] = np.arange(1, 24)  # Each joint's parent
        
        # Initialize joint positions
        if hasattr(smpl_model, 'J'):
            if hasattr(smpl_model.J, 'r'):
                # Chumpy object case
                self.J = np.array(smpl_model.J.r)
            else:
                # NumPy array case
                self.J = np.array(smpl_model.J)
        else:
            # If J is not directly available, we need to compute it
            # This is a simplified version - you might need to adjust based on your model
            self.J = np.zeros((24, 3))  # Assuming 24 joints, adjust if needed
        
        # Store the number of joints
        self.num_joints = self.J.shape[0]
        
        # Initialize with the current model state
        self.update_from_smpl()
    
    def update_from_smpl(self):
        """Update internal state from the SMPL model"""
        # Get the current vertices from the SMPL model
        if hasattr(self.smpl_model, 'r'):
            # Chumpy object case
            self.vertices = np.array(self.smpl_model.r)
        else:
            # Call forward method if available, or use directly if it's already the vertices
            if hasattr(self.smpl_model, 'forward'):
                # This is expected to return vertices directly
                self.vertices = self.smpl_model.forward(self.pose, self.betas)
            else:
                # Assume it's already the vertices
                self.vertices = np.array(self.smpl_model.v_template)
        
        # Store the current pose and shape parameters
        if hasattr(self.smpl_model.pose, 'r'):
            # Chumpy object case
            self.pose = np.array(self.smpl_model.pose.r)
            self.betas = np.array(self.smpl_model.betas.r)
        else:
            # NumPy array case
            self.pose = np.array(self.smpl_model.pose)
            self.betas = np.array(self.smpl_model.betas)
        
        # Update joint positions if available
        if hasattr(self.smpl_model, 'J'):
            if hasattr(self.smpl_model.J, 'r'):
                # Chumpy object case
                self.J = np.array(self.smpl_model.J.r)
            else:
                # NumPy array case
                self.J = np.array(self.smpl_model.J)
    
    def forward(self, pose=None, betas=None, **kwargs):
        """
        Forward pass for SMPL model
        
        Args:
            pose: pose parameters
            betas: shape parameters
            
        Returns:
            Updated vertices
        """
        if pose is not None:
            # Handle the case where pose is a Ch object
            if hasattr(pose, 'r'):
                pose_array = np.array(pose.r)
            else:
                pose_array = np.array(pose)
            
            # Assign to the SMPL model's pose
            # Handle both Chumpy and NumPy models
            if hasattr(self.smpl_model.pose, 'r'):
                self.smpl_model.pose[:] = pose_array
            else:
                self.smpl_model.pose = pose_array
            self.pose = pose_array
        
        if betas is not None:
            # Handle dimension mismatch
            if hasattr(self.smpl_model.betas, 'r'):
                expected_dim = len(self.smpl_model.betas.r)
            else:
                expected_dim = len(self.smpl_model.betas)
                
            if len(betas) != expected_dim:
                print(f"WARNING: Shape parameter dimension mismatch. Got {len(betas)}, expected {expected_dim}.")
                # Simple padding/truncation to match dimensions
                padded_betas = np.zeros(expected_dim, dtype=np.float64)
                min_dim = min(len(betas), expected_dim)
                padded_betas[:min_dim] = betas[:min_dim]
                betas = padded_betas
            
            # Handle the case where betas is a Ch object
            if hasattr(betas, 'r'):
                betas_array = np.array(betas.r)
            else:
                betas_array = np.array(betas)
                
            # Assign to SMPL model's betas
            # Handle both Chumpy and NumPy models
            if hasattr(self.smpl_model.betas, 'r'):
                self.smpl_model.betas[:] = betas_array
            else:
                self.smpl_model.betas = betas_array
            self.betas = betas_array
        
        # Update the vertices
        # If the model has a direct forward method, use it
        if hasattr(self.smpl_model, 'forward'):
            self.vertices = self.smpl_model.forward(pose=self.pose, betas=self.betas)
        elif hasattr(self.smpl_model, 'r'):
            # For Chumpy models, 'r' gives the current vertices
            self.vertices = np.array(self.smpl_model.r)
        else:
            # Fallback, though this shouldn't typically happen
            print("WARNING: No way to get updated vertices from SMPL model.")
            self.vertices = np.array(self.smpl_model.v_template)
        
        # Update joint positions if available
        if hasattr(self.smpl_model, 'J'):
            if hasattr(self.smpl_model.J, 'r'):
                self.J = np.array(self.smpl_model.J.r)
            else:
                self.J = np.array(self.smpl_model.J)
        
        return self.vertices
    
    def set_shape_family(self, family_id):
        """
        Set the shape family for the model
        
        Args:
            family_id: Animal family ID
        """
        # Load the family clusters data
        import os
        import pickle
        
        data_path = "./data/smal_CVPR2017_data.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            if 'cluster_means' in data and len(data['cluster_means']) > family_id:
                self.smpl_model.betas[:] = data['cluster_means'][family_id]
                self.betas = np.array(self.smpl_model.betas.r)
                self.update_from_smpl()
                print(f"Applied family shape parameters for family {family_id}")
    
    def initialize_for_cow(self):
        """Initialize the model specifically for cow visualization"""
        self.set_shape_family(3)  # 3 is the bovidae/cow family ID
        
    def get_transformed_joints(self, rotation_angle_x=0, rotation_angle_y=0, rotation_angle_z=0):
        """
        Get joint positions transformed by the current pose and shape
        
        Args:
            rotation_angle_x: Additional rotation angle around X axis in degrees (not used)
            rotation_angle_y: Additional rotation angle around Y axis in degrees (not used)
            rotation_angle_z: Additional rotation angle around Z axis in degrees (not used)
            
        Returns:
            numpy.ndarray: Transformed joint positions
        """
        # Try to get the transformed joints from the SMPL model
        if hasattr(self.smpl_model, 'J_transformed') and hasattr(self.smpl_model.J_transformed, 'r'):
            # Use the transformed joints directly from the model
            joints = np.array(self.smpl_model.J_transformed.r)
        elif hasattr(self.smpl_model, 'J') and hasattr(self.smpl_model.J, 'r'):
            # Use the base joints if transformed joints aren't available
            joints = np.array(self.smpl_model.J.r)
        else:
            # Fall back to our stored joints
            joints = self.J
        
        # Return the joints without any rotation
        # This should match how the mesh is displayed
        return joints 