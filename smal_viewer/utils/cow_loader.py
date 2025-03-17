import os
import pickle
import numpy as np
from pathlib import Path

def load_cow_model(file_path):
    """
    Load pre-registered cow model parameters from a pickle file
    
    Args:
        file_path: Path to the cow model pickle file
        
    Returns:
        dict: Dictionary containing cow model parameters
    """
    print(f"Loading cow model from: {file_path}")
    with open(file_path, 'rb') as f:
        cow_params = pickle.load(f, encoding='latin1')
    
    print(f"Loaded cow model with keys: {list(cow_params.keys())}")
    return cow_params

def apply_cow_params_to_model(model, cow_params):
    """
    Apply cow parameters to the SMAL model
    
    Args:
        model: SMAL model
        cow_params: Cow parameters from PKL file
    
    Returns:
        Updated SMAL model
    """
    # Extract shape parameters - check for both 'betas' and 'beta'
    betas = cow_params.get('betas', None)
    if betas is None:
        betas = cow_params.get('beta', None)  # Try singular form
        
    pose = cow_params.get('pose', None)
    
    if betas is not None:
        print(f"Applying cow shape parameters: {betas}")
        # Ensure betas is a numpy array
        if not isinstance(betas, np.ndarray):
            betas = np.array(betas, dtype=np.float64)
        
        # Apply pose if available
        if pose is not None and not isinstance(pose, np.ndarray):
            pose = np.array(pose, dtype=np.float64)
        
        # Update the model using the forward method
        model.forward(pose=pose, betas=betas)
    else:
        print("No shape parameters found in cow model!")
        
        # Try to load the SMAL data file and use bovidae (cow) family parameters
        try:
            project_root = get_project_root()
            possible_data_paths = [
                project_root / "data" / "smal_CVPR2017_data.pkl",
                project_root / "smal_CVPR2017_data.pkl",
                Path("./data/smal_CVPR2017_data.pkl"),
                Path("./smal_CVPR2017_data.pkl"),
            ]
            
            data_path = None
            for path in possible_data_paths:
                if path.exists():
                    data_path = path
                    break
            
            if data_path:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Bovidae (cows) are family 3
                if 'cluster_means' in data and len(data['cluster_means']) > 3:
                    cow_betas = data['cluster_means'][3]
                    print(f"Using bovidae family shape parameters from SMAL data")
                    model.forward(pose=pose or np.zeros_like(model.pose), betas=cow_betas)
        except Exception as e:
            print(f"Error loading SMAL data for cow parameters: {e}")
    
    return model

def get_available_cow_models(directory=None):
    """
    Get a list of available cow model files
    
    Args:
        directory: Directory to search for cow model files (default: data/cows)
        
    Returns:
        list: List of cow model file paths
    """
    if directory is None:
        # Try to find cow models in common locations
        from .loader import get_project_root
        project_root = get_project_root()
        possible_dirs = [
            project_root / "data" / "cows",
            project_root / "cows",
            Path("./data/cows"),
            Path("./cows"),
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                directory = dir_path
                break
    
    if directory is None or not os.path.exists(directory):
        print(f"No cow model directory found")
        return []
    
    # Find all .pkl files in the directory
    cow_files = []
    for file in os.listdir(directory):
        # Check that file doesn't start with a period and ends with .pkl
        if not file.startswith('.') and file.lower().endswith(".pkl") and "cow" in file.lower():
            cow_files.append(os.path.join(directory, file))
    
    print(f"Found {len(cow_files)} cow model files: {cow_files}")
    return cow_files 