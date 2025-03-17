import sys
import os
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# Import SMPL package for model evaluation
try:
    from SMPL.smpl_webuser.serialization import load_model
    HAS_SMPL = True
except ImportError:
    HAS_SMPL = False
    print("Warning: SMPL package not found. Falling back to native implementation.")

from .utils.loader import load_smal_model, load_animation_sequence
from .utils.cow_loader import load_cow_model, apply_cow_params_to_model
from .gui.viewer import MainWindow

def main():
    parser = argparse.ArgumentParser(description="SMAL Cow Model Viewer")
    parser.add_argument("--model", type=str, help="Path to the SMAL model file")
    # We can keep the family argument for backward compatibility but ignore it
    parser.add_argument("--family", type=int, default=3, help="Animal family ID (default: 3 for cows)")
    parser.add_argument("--animation", type=str, help="Path to animation sequence file")
    parser.add_argument("--cow", type=str, help="Path to cow model file")
    parser.add_argument("--use-smpl", action="store_true", help="Use SMPL package for model evaluation")
    
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow()
    
    # Determine which model evaluation method to use
    use_smpl = args.use_smpl and HAS_SMPL

    
    if use_smpl:
        print("Using SMPL package for model evaluation")
        # Load the model using SMPL's load_model function
        model_path = args.model or "./data/smal_CVPR2017.pkl"
        model = load_model(model_path)
        
        # Set the model to cow family (bovidae)
        # Load the family clusters data
        data_path = "./data/smal_CVPR2017_data.pkl"
        if os.path.exists(data_path):
            import pickle
            with open(data_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Set to bovidae (cow) family - index 3
            if 'cluster_means' in data and len(data['cluster_means']) > 3:
                model.betas[:] = data['cluster_means'][3]
                model.pose[:] = 0.
                model.trans[:] = 0.
                print("Applied bovidae family shape parameters from SMAL data")
        
        # Create a wrapper for the SMPL model to make it compatible with our viewer
        from .models.smpl_wrapper import SMPLWrapper
        model = SMPLWrapper(model)     
        
    else:
        # Use our native implementation
        print("Using native implementation for model evaluation")
        # Load SMAL model - always use family 3 (cow)
        model = load_smal_model(args.model, 3)
        
        # Initialize the model specifically for cows
        if hasattr(model, 'initialize_for_cow'):
            model.initialize_for_cow()
    
    # Load cow model if specified
    cow_path = None
    if args.cow and os.path.exists(args.cow):
        cow_path = args.cow
        cow_params = load_cow_model(args.cow)
        model = apply_cow_params_to_model(model, cow_params)
    
    # Load animation sequence
    animation_frames = load_animation_sequence(args.animation)
    
    # Set model in viewer
    window.set_model(model, animation_frames, args.model, args.animation)
    
    # If a cow model was specified, make sure it's in the cow_models list
    if cow_path:
        if cow_path not in window.cow_models:
            window.cow_models.append(cow_path)
            window.cow_combo.addItem(os.path.basename(cow_path))
            # Select the newly added cow model
            window.cow_combo.setCurrentIndex(window.cow_combo.count() - 1)
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 