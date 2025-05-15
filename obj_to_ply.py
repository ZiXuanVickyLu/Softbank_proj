import open3d as o3d
import numpy as np
import os
import argparse

def convert_obj_to_ply(obj_file_path, ply_file_path, use_vertices_directly=True):
    """
    Converts an OBJ file to a PLY file that can be loaded with the load_point_cloud function.
    
    Args:
        obj_file_path (str): Path to the input OBJ file.
        ply_file_path (str): Path to save the output PLY file.
        use_vertices_directly (bool): If True, use the mesh vertices directly instead of sampling.
    """
    try:
        # Read the triangle mesh from the OBJ file
        print(f"Loading OBJ from {obj_file_path}...")
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        
        # Check if the mesh was loaded successfully
        if not mesh.has_vertices():
            print(f"Error: Could not read mesh from {obj_file_path}")
            return False

        print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Create a point cloud
        pcd = o3d.geometry.PointCloud()
        
        if use_vertices_directly:
            # Use vertices directly without sampling
            pcd.points = mesh.vertices
            print(f"Using all {len(mesh.vertices)} vertices directly")
        else:
            # Sample points from the mesh surface
            if mesh.has_triangles():
                number_of_points = 2048  # Default sample size, you can adjust
                pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
                print(f"Sampled {number_of_points} points uniformly from mesh surface")
            else:
                # If there are no triangles, fall back to vertices
                pcd.points = mesh.vertices
                print(f"No triangles found. Using {len(mesh.vertices)} vertices")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(ply_file_path)), exist_ok=True)
        
        # Save the point cloud as a PLY file
        o3d.io.write_point_cloud(ply_file_path, pcd)
        print(f"Successfully saved point cloud to {ply_file_path}")
        return True

    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OBJ file to PLY point cloud")
    parser.add_argument("--input", "-i", required=True, help="Input OBJ file path")
    parser.add_argument("--output", "-o", required=True, help="Output PLY file path")
    parser.add_argument("--sample", "-s", action="store_true", help="Sample points instead of using vertices directly")
    
    args = parser.parse_args()
    
    success = convert_obj_to_ply(
        args.input, 
        args.output, 
        use_vertices_directly=not args.sample
    )
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!") 