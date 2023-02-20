from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import os
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt



save_root = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/sparse/sim_stereo'
root_dir = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/data/nyudepthv2/train'
subject = 'bedroom_0004'
sub_dir = os.path.join(root_dir,subject)
save_dir = os.path.join(save_root,subject)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# define sparsifier
cmap = plt.cm.viridis
sparsifier = SimulatedStereo(num_samples=10000)

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

for file in os.listdir(sub_dir):

    # h5 path and file name
    file_path = os.path.join(sub_dir,file)
    file_name = file.split('.h5')[0]

    # get rgb and depth from h5
    h5f = h5py.File(file_path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])

    # create sparse depth map with 100 samples
    mask_keep = sparsifier.dense_to_sparse(rgb, depth)
    sparse_depth = np.zeros(depth.shape)
    sparse_depth[mask_keep] = depth[mask_keep]

    # create rgb for depth map
    sparse_depth_color = colored_depthmap(sparse_depth)
   
    sparse_depth_color = Image.fromarray(sparse_depth_color.astype('uint8'),'RGB')
    sparse_depth_color.save(os.path.join(save_dir,f'{file_name}-sp10000_depth.jpg'), "JPEG")




