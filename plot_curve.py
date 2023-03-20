import matplotlib.pyplot as plt
import csv 
import pandas as pd
import numpy as np

train = pd.read_csv('/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/results/nyudepthv2.sparsifier=uar.samples=0.modality=rgb.arch=resnet18.decoder=deconv3.criterion=l1.lr=0.01.bs=8.pretrained=False/train.csv')
test = pd.read_csv('/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/results/nyudepthv2.sparsifier=uar.samples=0.modality=rgb.arch=resnet18.decoder=deconv3.criterion=l1.lr=0.01.bs=8.pretrained=False/test.csv')

train = train[train.columns[:6]]
test = test[test.columns[:6]]

figure, axis = plt.subplots(2,3, figsize=(20,10))

# axis[0,0].plot(np.arange(15),train['mse'])
# axis[0,0].set_title("Training MSE")

# axis[0,1].plot(np.arange(15),train['rmse'])
# axis[0,1].set_title("Training RMSE")

# # axis[0,2].plot(np.arange(15),train['absrel'])
# # axis[0,2].set_title("Training AbsRel")

# axis[1,0].plot(np.arange(15),train['lg10'])
# axis[1,0].set_title("Training Log10")

# axis[1,1].plot(np.arange(15),train['mae'])
# axis[1,1].set_title("Training MAE")

# axis[1,2].plot(np.arange(15),train['delta1'])
# axis[1,2].set_title("Training Delta")

# plt.savefig("/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/results/nyudepthv2.sparsifier=uar.samples=0.modality=rgb.arch=resnet18.decoder=deconv3.criterion=l1.lr=0.01.bs=8.pretrained=False/train_acc.png")

axis[0,0].plot(np.arange(15),test['mse'])
axis[0,0].set_title("Testing MSE")

axis[0,1].plot(np.arange(15),test['rmse'])
axis[0,1].set_title("Testing RMSE")

axis[0,2].plot(np.arange(15),test['absrel'])
axis[0,2].set_title("Testing AbsRel")

axis[1,0].plot(np.arange(15),test['lg10'])
axis[1,0].set_title("Testing Log10")

axis[1,1].plot(np.arange(15),test['mae'])
axis[1,1].set_title("Testing MAE")

axis[1,2].plot(np.arange(15),test['delta1'])
axis[1,2].set_title("Testing Delta")

plt.savefig("/nfs/masi/zhouy26/ml-proj/sparse-to-dense.pytorch/results/nyudepthv2.sparsifier=uar.samples=0.modality=rgb.arch=resnet18.decoder=deconv3.criterion=l1.lr=0.01.bs=8.pretrained=False/test_acc.png")
