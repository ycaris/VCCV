import numpy as np
import h5py
import os

mat_path = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/data/nyu_depth_v2_labeled.mat'
f = h5py.File(mat_path, 'r')

# ref_path = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/data/nyudepthv2/train/basement_0001a/00001.h5'
# f2 = h5py.File(ref_path, 'r')

output_dir = '/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/data/nyudepthv2-seg'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1449):  # loop through 1449 images

    rgbnp = np.array(f['images'][i])
    rgbnp = np.transpose(rgbnp, (0, 2, 1))

    depnp = np.array(f['depths'][i])
    depnp = np.transpose(depnp, (1, 0))

    segnp = np.array(f['labels'][i])
    segnp = np.transpose(segnp, (1, 0))

    h5_name = os.path.join(output_dir, f'{i:04d}.h5')

    with h5py.File(h5_name, 'w') as f2:
        f2.create_dataset('rgb', data=rgbnp)
        f2.create_dataset('depth', data=depnp)
        f2.create_dataset('labels', data=segnp)
    f2.close()
    print(f'finished {i}')

# print(f.keys())
# print(f2.keys())
# print(f['#refs#'])
# print(f['rawRgbFilenames'][0])

# rgbnp2 = np.array(f2['rgb'])
# depthnp2 = np.array(f2['depth'])
# # rgbnp2 = np.transpose(rgbnp2, (0, 2, 1))
# print(rgbnp2.shape)
# print(depthnp2.shape)
# __path__
