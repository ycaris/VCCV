import h5py
import os
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

# root_dir = '/data/maiziezhou_lab/Jamiezhou/ml_project/KITTI_test/depth_selection/val_selection_cropped/groundtruth_depth/'
# save_dir = '/data/maiziezhou_lab/Jamiezhou/ml_project/KITTI_test/depth_selection/val_selection_cropped/rgb/'
# root_dir = '/data/maiziezhou_lab/Jamiezhou/ml_project/Sparse-Depth-Completion/Saved/results/'
# save_dir = '/data/maiziezhou_lab/Jamiezhou/ml_project/Sparse-Depth-Completion/Saved/rgb/'
# root_dir = '/data/maiziezhou_lab/Jamiezhou/ml_project/crop_KITTI/proj_depth/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/'
# save_dir = '/scratch/zhouz19/visualizaiton/actual/'
root_dir = "/scratch/zhouz19/Sparse-Depth-Completion/Data_test/depth_selection/val_selection_cropped/groundtruth_depth/"
save_dir = "/scratch/zhouz19/Sparse-Depth-Completion/Saved/best_model/ground_truth/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cmap = plt.cm.viridis

np.set_printoptions(threshold=sys.maxsize)

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def depth_read(img, sparse_val):
    depth_png = np.array(img, dtype=int)
    # depth_png = np.expand_dims(depth_png, axis=2)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = sparse_val
    return depth

for file in os.listdir(root_dir):
    # h5 path and file name
    file_path = os.path.join(root_dir,file)
    file_name = file.split('.png')[0]

    depth = Image.open(file_path)
    depth = depth_read(depth, 0)

    # create rgb for depth map
    depth_color = colored_depthmap(depth)

    # # save image and rgb as jpeg
    depth_color = Image.fromarray(depth_color.astype('uint8'),'RGB')
    depth_color.save(os.path.join(save_dir,f'{file_name}-depth.jpg'), "JPEG")