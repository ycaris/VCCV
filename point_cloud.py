import os
import open3d as o3d
import numpy as np
import h5py



# Depth camera parameters:
FX_DEPTH = 5.8262448167737955e+02
FY_DEPTH = 5.8269103270988637e+02
CX_DEPTH = 3.1304475870804731e+02
CY_DEPTH = 2.3844389626620386e+02

depth_path = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/predict_depth/3.5/pred/01397.npy'
depth_image = np.load(os.path.join(depth_path))
    


# compute point cloud:
pcd = []
height, width = depth_image.shape
depth_image = depth_image.reshape(height,width)

for i in range(height):
    for j in range(width):
        z = depth_image[i][j]
        x = (j - CX_DEPTH) * z / FX_DEPTH
        y = (i - CY_DEPTH) * z / FY_DEPTH
        pcd.append([x, y, z])
pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
# Visualize:
o3d.visualization.draw_geometries([pcd_o3d])


# file_path = "/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/predict_depth/3.5/target/01397.npy"
# depth = np.load(os.path.join(file_path))

# # compute point cloud:
# pcd = []
# height, width = depth.shape
# depth_image = depth.reshape(height,width)

# for i in range(height):
#     for j in range(width):
#         z = depth_image[i][j]
#         x = (j - CX_DEPTH) * z / FX_DEPTH
#         y = (i - CY_DEPTH) * z / FY_DEPTH
#         pcd.append([x, y, z])
# pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
# pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
# # Visualize:
# o3d.visualization.draw_geometries([pcd_o3d])

