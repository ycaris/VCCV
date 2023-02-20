import dataloaders.transforms as trans
import numpy as np
import h5py
from PIL import Image
import os

save_dir = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/visual/tran_bed0004-00001'

img = "/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/data/nyudepthv2/train/bedroom_0004/00001.h5"

h5f = h5py.File(img, "r")
rgb = np.array(h5f['rgb'])
rgb = np.transpose(rgb, (1, 2, 0))
img = Image.fromarray(rgb.astype('uint8'), 'RGB')

img1 = trans.adjust_hue(img,0)
img1.save(os.path.join(save_dir,'hue-0.0.jpg'), "JPEG")

img2 = trans.adjust_hue(img,-0.5)
img2.save(os.path.join(save_dir,'hue--0.5.jpg'), "JPEG")

img3 = trans.adjust_hue(img,0.3)
img3.save(os.path.join(save_dir,'hue-+0.3.jpg'), "JPEG")

img4 = trans.adjust_hue(img,0.5)
img4.save(os.path.join(save_dir,'hue-+0.5.jpg'), "JPEG")

img5 = trans.adjust_saturation(img,0.5)
img5.save(os.path.join(save_dir,'sat-0.5.jpg'), "JPEG")

img6 = trans.adjust_saturation(img,1.5)
img6.save(os.path.join(save_dir,'sat-1.5.jpg'), "JPEG")

img7 = trans.adjust_saturation(img,1.0)
img7.save(os.path.join(save_dir,'sat-1.0.jpg'), "JPEG")

img8 = trans.adjust_saturation(img,2.0)
img8.save(os.path.join(save_dir,'sat-2.0.jpg'), "JPEG")
