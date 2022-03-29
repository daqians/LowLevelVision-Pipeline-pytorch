# Usage:
# https://github.com/jacobgil/pytorch-grad-cam
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.ipynb
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Semantic%20Segmentation.ipynb
#
# For Transformers:
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
# https://www.kaggle.com/piantic/vision-transformer-vit-visualize-attention-map
# https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb

# This service is for Gradient-weighted Class Activation Mapping (Grad-CAM).
# Uses the gradients of any target concept (say ‘dog’ in a classification network
# or a sequence of words in captioning network) flowing into the final convolutional
# layer to produce a coarse localization map highlighting the important regions
# in the image for predicting the concept.

import sys
sys.path.append('G:\BACKUP\learning\programs\OracleRecognition\Low_level_CV_PPL')

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50
from models.model_RCRN import GeneratorUNet, Discriminator, weights_init_normal

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import requests


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def loadModel():
    generator = GeneratorUNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load("../saved_models/Dataset1/model_RCRN_generator_50.pth", map_location=device)
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    return generator


def input_data(ImgName):
    rgb_img = Image.open('../data/Dataset2/train/%s' % ImgName).convert('RGB')
    print(rgb_img.size)
    img = rgb_img.copy()

    rgb_img = np.array(rgb_img.resize((256, 256)))
    rgb_img = np.float32(rgb_img) / 255

    transforms_ = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]
    transform = transforms.Compose(transforms_)
    input_tensor = transform(img).unsqueeze(0)

    return rgb_img, input_tensor


# 加载自己的模型 - low level CV ??
def low_level_CV_example(ImgName):
    # 模型参数加载，以及确定要提取的模型layer，构建一个hook
    model = loadModel()
    # print(generator)
    for module in model.up7.children():
        for mm in module.children():
            if "InstanceNorm2d" in mm._get_name():
                target_layers = [mm]
    # target_layers = [generator.final]

    for name, module in model.named_modules():
        print(name)
        print(module)

    # 输入数据
    rgb_img, input_tensor = input_data(ImgName)
    model = model.eval()
    model = model.cuda()
    input_tensor = input_tensor.cuda()
    output = model(input_tensor)
    print(type(output))

    # low-level CV 的输出
    targets = Image.open('../data/Dataset2/target/%s' % ImgName).convert('RGB')
    targets = np.array(targets.resize((256, 256)))
    targets = np.float32(targets) / 255

    # 进行CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization


# 加载官方模型 - CV classification
def resnet_classification_example(ImgName):
    # 模型参数加载，以及确定要提取的模型layer，构建一个hook
    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]

    # 模型输入，以及输出的数据结构
    rgb_img, input_tensor = input_data(ImgName)
    # model = model.eval()
    # model = model.cuda()
    # input_tensor = input_tensor.cuda()
    # output = model(input_tensor)
    # print(type(output))

    # classification 的输出
    targets = [ClassifierOutputTarget(281)]
    print(type(targets))

    # 进行CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

# 加载官方模型 - CV Semantic Segmentation
def resnet_semanticSegmentation(ImgName):
    rgb_img, input_tensor = input_data(ImgName)

    # image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
    # image = np.array(Image.open(requests.get(image_url, stream=True).raw))
    # rgb_img = np.float32(image) / 255
    # input_tensor = preprocess_image(rgb_img,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    model = deeplabv3_resnet50(pretrained=True, progress=False)
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # output = model(input_tensor)
    # print(type(output), output.keys())


    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    car_category = sem_class_to_idx["car"]
    car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    car_mask_float = np.float32(car_mask == car_category)
    # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
    # both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))

    target_layers = [model.model.backbone.layer4]
    targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
# 加载官方模型 - Transformer for classification task
def Transformer_classification(ImgName):
    rgb_img, input_tensor = input_data(ImgName)

    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=True)
    model.eval()
    model = model.cuda()

    target_layers = [model.blocks[-1].norm1]

    cam = GradCAM(model=model,
                               target_layers=target_layers,
                               use_cuda=True,
                               reshape_transform=reshape_transform,
                               ablation_layer=AblationLayerVit)
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=None,
                        aug_smooth=None)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    return cam_image


if __name__ == '__main__':
    name = "oc_01_1_0003_1_13.png"
    visualization = resnet_classification_example(name)
    # visualization = low_level_CV_example(name)
    # visualization = resnet_semanticSegmentation(name)
    # visualization = Transformer_classification(name)

    Image.fromarray(visualization).save("../data/Dataset2/attention_map/%s" % name)
