import torch
import numpy as np
import open3d as o3d
from pytorch3d.ops import chamfer_distance
from pytorch3d.transforms import so3_exponential_map
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return torch.tensor(np.asarray(pcd.points), dtype=torch.float32).to(device)

def visualize_step(source, target, transformed, step):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source.cpu().numpy())
    source_pcd.paint_uniform_color([1, 0, 0])  

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.cpu().numpy())
    target_pcd.paint_uniform_color([0, 1, 0])  

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed.cpu().detach().numpy())
    transformed_pcd.paint_uniform_color([0, 0, 1])  

    print(f"Step {step}: Visualizing...")
    o3d.visualization.draw_geometries([source_pcd, target_pcd, transformed_pcd])

source_pcd = load_point_cloud("source.ply")  
target_pcd = load_point_cloud("target.ply")

rot_vector = torch.zeros(3, requires_grad=True, device=device)  
translation = torch.zeros(3, requires_grad=True, device=device) 

optimizer = torch.optim.Adam([rot_vector, translation], lr=0.01)

num_iterations = 100
for i in range(num_iterations):
    optimizer.zero_grad()
    
    
    R = so3_exponential_map(rot_vector.unsqueeze(0)).squeeze(0)  
    
    transformed_pcd = (source_pcd @ R.T) + translation  
    
    loss, _ = chamfer_distance(transformed_pcd.unsqueeze(0), target_pcd.unsqueeze(0))
    
    loss.backward()
    optimizer.step()
    
    print(f"Iteration {i}: Loss = {loss.item():.6f}")

    if i % 10 == 0:
        visualize_step(source_pcd, target_pcd, transformed_pcd, i)

visualize_step(source_pcd, target_pcd, transformed_pcd, "Final")
