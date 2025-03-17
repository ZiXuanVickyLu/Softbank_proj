import os
import numpy as np
from pathlib import Path
from ..models.smal_model import SMALModel
import pickle
import torch

def get_project_root():
    """Get the absolute path to the project root directory"""
    # The module file is in smal_viewer/utils/loader.py
    # So we need to go up two levels to get to the project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root

def load_smal_model(model_path=None, shape_family_id=1):
    """
    Load the SMAL model from the specified path
    
    Args:
        model_path: Path to the SMAL model pickle file
        shape_family_id: Animal family ID (default: 1 for dogs)
                         0-felidae(cats); 1-canidae(dogs); 2-equidae(horses);
                         3-bovidae(cows); 4-hippopotamidae(hippos)
    
    Returns:
        SMALModel: The loaded SMAL model
    """
    project_root = get_project_root()
    
    if model_path is None:
        # Try to find the model in common locations
        possible_paths = [
            project_root / "data" / "smal_CVPR2017.pkl",
            project_root / "smal_CVPR2017.pkl",
            Path("./data/smal_CVPR2017.pkl"),
            Path("./smal_CVPR2017.pkl"),
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = path
                print(f"Found model at: {path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "SMAL model file not found. Please download the SMAL model and "
                "specify the path to the model file. Expected locations:\n" +
                "\n".join(str(p) for p in possible_paths)
            )
    
    # Also try to find the SMAL data file
    data_path = None
    possible_data_paths = [
        project_root / "data" / "smal_CVPR2017_data.pkl",
        project_root / "smal_CVPR2017_data.pkl",
        Path("./data/smal_CVPR2017_data.pkl"),
        Path("./smal_CVPR2017_data.pkl"),
    ]
    
    for path in possible_data_paths:
        if path.exists():
            data_path = path
            print(f"Found SMAL data at: {path}")
            break
    
    print(f"Loading model from: {model_path} with family ID: {shape_family_id}")
    model = SMALModel(model_path, shape_family_id)
    
    # If we found the data file, apply the family-specific shape parameters
    if data_path:
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            if 'cluster_means' in data and len(data['cluster_means']) > shape_family_id:
                # Apply the family-specific shape parameters
                family_betas = data['cluster_means'][shape_family_id]
                print(f"Applying family-specific shape parameters for family {shape_family_id}")
                model.forward(pose=np.zeros_like(model.pose), betas=family_betas)
        except Exception as e:
            print(f"Error loading SMAL data: {e}")
    
    return model

def load_animation_sequence(sequence_path=None):
    """
    Load an animation sequence for the SMAL model
    
    Args:
        sequence_path: Path to the animation sequence file
        
    Returns:
        list: List of pose parameters for each frame
    """
    # This is a placeholder - in a real implementation, you would load
    # animation data from a file. For now, let's create a simple animation.
    
    # Create a simple walking animation (rotating the legs)
    num_frames = 60
    poses = []
    
    # SMAL has 35 joints, each with 3 rotation parameters
    base_pose = np.zeros(35 * 3)
    
    for i in range(num_frames):
        pose = base_pose.copy()
        
        # Animate front legs (joints 3 and 4)
        angle = np.sin(i * 2 * np.pi / num_frames) * 0.3
        pose[3*3+1] = angle  # Left front leg
        pose[4*3+1] = -angle  # Right front leg
        
        # Animate back legs (joints 5 and 6)
        pose[5*3+1] = -angle  # Left back leg
        pose[6*3+1] = angle  # Right back leg
        
        poses.append(pose)
    
    print(f"Created animation sequence with {len(poses)} frames")
    return poses

def load_model_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Add debugging to check the shape parameters
    print("Shape parameters in PKL:", data.get('betas', None))
    print("Pose parameters in PKL:", data.get('pose', None))
    
    # Make sure parameters are properly formatted
    betas = data.get('betas', None)
    pose = data.get('pose', None)
    
    if betas is not None and not isinstance(betas, np.ndarray):
        betas = np.array(betas, dtype=np.float64)
    
    if pose is not None and not isinstance(pose, np.ndarray):
        pose = np.array(pose, dtype=np.float64)
    
    return data, betas, pose 