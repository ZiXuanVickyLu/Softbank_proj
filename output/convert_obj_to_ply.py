import open3d as o3d
import numpy as np

def convert_obj_to_ply(obj_file_path, ply_file_path, number_of_points=2048):
    """
    Loads an OBJ file, samples points from its mesh, and saves it as a PLY file.

    Args:
        obj_file_path (str): Path to the input OBJ file.
        ply_file_path (str): Path to save the output PLY file.
        number_of_points (int): Number of points to sample from the mesh.
    """
    try:
        # Read the triangle mesh from the OBJ file
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        
        # Check if the mesh was loaded successfully
        if not mesh.has_vertices():
            print(f"Error: Could not read mesh from {obj_file_path}")
            return

        # Sample points from the mesh
        # If the mesh is already a point cloud (e.g., vertices only, no faces),
        # you might want to use its vertices directly.
        # Here, we assume it's a mesh and sample points uniformly.
        if mesh.has_triangles():
            pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
        else:
            # If there are no triangles, use the vertices as points
            print("Warning: Mesh has no triangles. Using vertices as point cloud.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            # Optionally, if you want to ensure a specific number of points even from vertices:
            # if len(mesh.vertices) > number_of_points:
            #     indices = np.random.choice(len(mesh.vertices), number_of_points, replace=False)
            #     pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[indices])
            # elif len(mesh.vertices) < number_of_points:
            #     print(f"Warning: Number of vertices ({len(mesh.vertices)}) is less than desired points ({number_of_points}). Using all vertices.")


        # Save the point cloud as a PLY file
        o3d.io.write_point_cloud(ply_file_path, pcd)
        print(f"Successfully converted {obj_file_path} to {ply_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    input_obj_file = "render.obj"  # Replace with your OBJ file path
    output_ply_file = "render_500.ply" # Replace with your desired PLY file path
    #input_obj_file = "smal_target_cow_alph5.obj"  # Replace with your OBJ file path
    # output_ply_file = "smal_target_cow_alph5.ply" # Replace with your desired PLY file path
    # input_obj_file = "template_aligned.obj"  # Replace with your OBJ file path
    # output_ply_file = "template_aligned.ply" # Replace with your desired PLY file path
    # input_obj_file = "template.obj"  # Replace with your OBJ file path
    # output_ply_file = "template_500.ply" # Replace with your desired PLY file path
    num_points_to_sample = 500   # Number of points to sample from the mesh

    # --- Conversion ---
    print(f"Attempting to convert {input_obj_file} to {output_ply_file}...")
    convert_obj_to_ply(input_obj_file, output_ply_file, num_points_to_sample)

    # --- Verification (Optional) ---
    # You can try loading the generated PLY file to verify
    try:
        pcd_loaded = o3d.io.read_point_cloud(output_ply_file)
        if not pcd_loaded.has_points():
            print(f"Verification failed: Could not load points from {output_ply_file}")
        else:
            print(f"Verification successful: Loaded {len(pcd_loaded.points)} points from {output_ply_file}")
            # To use with your load_point_cloud function:
            # import torch
            # points_tensor = torch.tensor(np.asarray(pcd_loaded.points), dtype=torch.float32)
            # print(f"Shape of tensor for your load_point_cloud: {points_tensor.shape}")
    except Exception as e:
        print(f"Error during verification: {e}") 