import h5py
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root_dir = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/predict_depth/3.5/rgb'
subject = "rgb-png"
# dir = os.path.join(root_dir,subject)
dir = os.path.join(root_dir)
save_root_dir = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/predict_depth/3.5'
save_dir = os.path.join(save_root_dir,subject)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cmap = plt.cm.viridis

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

for file in os.listdir(dir):
    # # h5 path and file name
    file_path = os.path.join(dir,file)
    # file_name = file.split('.h5')[0]

    # # get rgb and depth from h5
    # h5f = h5py.File(file_path, "r")
    # rgb = np.array(h5f['rgb'])
    # rgb = np.transpose(rgb, (1, 2, 0))
    # depth = np.array(h5f['depth'])


    # # normalize depth map to 0 and 255
    # depth = ((depth - depth.min()) * (1/(depth.max() - depth.min()) * 255)).astype('uint8')

    # # create rgb for depth map
    # depth_color = colored_depthmap(depth)

    rgb = np.load(file_path)
    file_name = file.split('.npy')[0]

    # # save image and rgb as jpeg
    img = Image.fromarray(rgb.astype('uint8'), 'RGB')
    img.save(os.path.join(save_dir,f'{file_name}-img.jpg'), "JPEG")
    # depth_color = Image.fromarray(depth_color.astype('uint8'),'RGB')
    # depth_color.save(os.path.join(save_dir,f'{file_name}-depth.jpg'), "JPEG")
    
    print(f"finished {file}")