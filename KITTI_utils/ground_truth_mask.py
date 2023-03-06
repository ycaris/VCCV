import h5py
import os
from PIL import Image
import numpy as np
import sys
import numpy.ma as ma
import matplotlib.pyplot as plt

pred_dir = "/scratch/zhouz19/Sparse-Depth-Completion/Saved/pretrained/results/"
ground_dir = "/scratch/zhouz19/Sparse-Depth-Completion/Data_test/depth_selection/val_selection_cropped/groundtruth_depth"
save_dir = "/scratch/zhouz19/Sparse-Depth-Completion/Saved/pretrained/rgb/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cmap = plt.cm.viridis

np.set_printoptions(threshold=sys.maxsize)

def depth_read(filename, sparse_val):
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = sparse_val
    return depth

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

for file in os.listdir(ground_dir):
    # h5 path and file name
    try: 
        file_path = os.path.join(ground_dir,file)
        file_name = file.split('.png')[0]

        ground_depth = depth_read(file_path, 0)
        # print(ground_depth)

        pred_path = os.path.join(pred_dir, file.replace("groundtruth_depth", "velodyne_raw"))
        pred_depth = depth_read(pred_path, 0)
        pred_depth[np.where(ground_depth == 0)] = 0

        # create rgb for depth map
        depth_color = colored_depthmap(pred_depth)

        # # save image and rgb as jpeg
        depth_color = Image.fromarray(depth_color.astype('uint8'),'RGB')
        depth_color.save(os.path.join(save_dir,f'{file_name}-depth.jpg'), "JPEG")
    except:
        pass
    # print(f"finished {file}")