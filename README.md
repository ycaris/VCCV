# VCCV Depth Completion App

This repo contains the the VCCV website app for Depth Completion. The model is adpapted from the Sparse-Depth-Completion [Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty](https://arxiv.org/abs/1902.05356) by [Wouter Van Gansbeke](https://github.com/wvangansbeke), Davy Neven, Bert De Brabandere and Luc Van Gool. More information on how to run the network can be reference in [here](/Sparse-Depth-Completion-master/README.md).

```
@inproceedings{wvangansbeke_depth_2019,
    author={Van Gansbeke, Wouter and Neven, Davy and De Brabandere, Bert and Van Gool, Luc},
    booktitle={2019 16th International Conference on Machine Vision Applications (MVA)},
    title={Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty},
    year={2019},
    pages={1-6},
    organization={IEEE}
}
```
[link](https://drive.google.com/file/d/1Hpd9QV4CaXtqtys9yvqRFpOCjkOmI9EU/view?usp=share_link)

## Requirements
Python 3.7
The most important packages are pytorch, torchvision, numpy, pillow and matplotlib.
(Works with Pytorch 1.1)
Flask: Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
werkzeug.utils: secure_filename

## Dataset
This app processes data from the [Kitti dataset](www.cvlibs.net/datasets/kitti/), or other forms of data with 1216 * 325 dimensions.

## Run Code
The website is currently only availalbe on your local computer. To access the website, clone the repository, run app.py, and access http://localhost:5050

## Model
For more information about the Model, please see the read me under the Sparse-Depth-Completion-master folder.