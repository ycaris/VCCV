import os
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from subprocess import Popen


parser = ArgumentParser(description="visualize:")
parser.add_argument('--save_dir','-s', help="output path")
parser.add_argument('--root_dir','-r', help="input path")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

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
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(img, dtype=int)
    # depth_png = np.expand_dims(depth_png, axis=2)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float64) / 256.
    depth[depth_png == 0] = sparse_val
    return depth

for file in os.listdir(args.root_dir):
    # h5 path and file name
    file_path = os.path.join(args.root_dir,file)
    file_name = file.split('.png')[0]

    depth = Image.open(file_path)
    depth = depth_read(depth, 0)

    # create rgb for depth map
    depth_color = colored_depthmap(depth)

    # # save image and rgb as jpeg
    depth_color = Image.fromarray(depth_color.astype('uint8'),'RGB')
    depth_color.save(os.path.join(args.save_dir,file_name + '_depth.png'), "PNG")
