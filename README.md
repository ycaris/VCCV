# VCCV Depth Estimation Project
Exploration of Depth Estimation Techniques: Monocular Depth Estimation with pretrained MobileNetV2
This project implements a deep learning neural network model to generate a depth image from RGB image. The model used has a encoder-decoder architecture. We used a pretrained MobileNet for image classification task on the ImageNet. The decoder is several upsampling convolution layers. 


original video             |  depth output
:-------------------------:|:-------------------------:
![](https://github.com/HiroakiYo/Depth_Estimation_Project/blob/main/Demo/exmaples/movie_real.gif?raw=true)  |  ![](https://github.com/HiroakiYo/Depth_Estimation_Project/blob/main/Demo/exmaples/movie_depth.gif?raw=true)


This project was implemented by referencing to the following paper:


[High Quality Monocular Depth Estimation via Transfer Learning (arXiv 2018)](https://arxiv.org/abs/1812.11941) **Ibraheem Alhashim** and **Peter Wonka**

## How to use the code
All the training and evaluating code are stored in Jupyter Notebook files (.ipynb)

**Model is trained with "train_mobilenetv2.ipynb"**
- Our pretrained model is stored in ./model directoty, you can also train your model with the code as well.
- The Dataset used for training is NYU Depth V2 obtained from kaggle. More will be discussed below.

**Video and gif output is generated with "generate_depth_video.ipynb"**
- Make subdirectories for the ./Demo directory according to the usage of "generate_depth_video.ipynb"
- Run the entire notebook will generate a depth video output and two gif images outputs

## Dataset
[NYU Depth V2] (https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) - about 4 GB
- Unpack the dataset into ./data directory
- Configure the ./data directory structures according to the "train_mobilenetv2.ipynb" so that training can be done.
