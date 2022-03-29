# LowLevelVision-Pipeline-pytorch
This is an easy-handle pipeline project for common AI tasks by pytorch. Currently, this project is mainly about low level vision tasks (Image2Image)


### Files:
datasets.py: loading images from target path
train.py: Model training pipeline
predict.py: task processing pipeline

### Folders:
data: datasets for training and testing
models: different NN models
saved_models: trained models weights
validImages: validition of test images for visualization
util: Tools for utilizing the model training, or additional processing like visualization.
wandb: wandb recoder for result/parameter visualization (optional)


### Requirements:
+ CUDA==10.2
+ cudnn==8.0
+ python==3.6
+ torch==1.8.0
+ torchvision==0.9.0
+ DNN-printer==0.0.2
+ numpy
+ tqdm
+ wandb (optional)
+ argparse
+ wmi (optional)
