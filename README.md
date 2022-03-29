# LowLevelVision-Pipeline-pytorch
This is an easy-handle pipeline project for common AI tasks by pytorch. Currently, this project is mainly about low level vision tasks (Image2Image)


###Requirements:
  +CUDA==10.2
  +cudnn==8.0
  +python==3.6
  +torch==1.8.0
  +torchvision==0.9.0
  +DNN-printer==0.0.2
  +numpy
  +tqdm
  +wandb (optional)
  +argparse
  +wmi (optional)


Folders:
1. data: datasets for training and testing
2. models: different NN models
3. saved_models: trained models weights
4. validImages: validition of test images for visualization
5. util: Tools for utilizing the model training, or additional processing like visualization.
6. wandb: wandb recoder for result/parameter visualization (optional)
