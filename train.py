##### External Interface #####
import argparse
import time
import datetime
import sys
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid

##### Internal Interface #####
from util.LossFunctions import VGGPerceptualLoss, SK_loss
from models.model_RCRN import GeneratorUNet, Discriminator, weights_init_normal
from datasets import *
from util.TestMetrics import get_PSNR, get_SSIM

##### Optional Tools #####
from DNN_printer import DNN_printer
import wandb
import wmi


def TrainModel(opt):
    ##### Make necessary directories #####
    os.makedirs("validImages/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    ##### Determine CUDA #####
    cuda = True if torch.cuda.is_available() else False
    # device = torch.device("cuda" if cuda else "cpu")

    ##### Initialize models: generator and discriminator #####
    generator = GeneratorUNet(in_channels=opt.channels, out_channels= opt.channels)
    discriminator = Discriminator(in_channels=opt.channels)
    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(
            torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    ##### Print Model Size #####
    DNN_printer(generator, (opt.channels, opt.img_height, opt.img_width), opt.batch_size)

    ##### Define Optimizers #####
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    ##### Configure Dataloaders #####
    # data transformations
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]

    # training data
    dataloader = DataLoader(
        ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, file_set="separate",
                     root2="data/%s/target" % opt.dataset_name),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # validating data
    val_dataloader = DataLoader(
        ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, mode="val", file_set="separate",
                     root2="data/%s/target" % opt.dataset_name),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    ########################################
    #####      Parameter Setting       #####
    ########################################

    ##### Loss functions #####
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss()

    if cuda:
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    ##### Loss weight of L1 pixel-wise loss between translated image and real image #####
    lambda_GAN = 1
    lambda_pixel = 100
    lambda_vgg = 0.05

    ##### Calculate output of image discriminator #####
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ########################################
    #####      Tools Definitions       #####
    ########################################

    ##### calculate the evaluation metrics on validation set #####
    def output_metrics():
        imgs = next(iter(val_dataloader))
        input = Variable(imgs["B"].type(Tensor))
        target = Variable(imgs["A"].type(Tensor))
        output = generator(input)

        ndarr_target = make_grid(target.data).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                    torch.uint8).numpy()
        ndarr_output = make_grid(output.data).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                    torch.uint8).numpy()

        PSNR = get_PSNR(ndarr_target, ndarr_output)
        SSIM = get_SSIM(ndarr_target, ndarr_output)

        return PSNR, SSIM

    ##### Saves a generated sample from the validation set #####
    def sample_images(batches_done):
        imgs = next(iter(val_dataloader))
        input = Variable(imgs["B"].type(Tensor))
        target = Variable(imgs["A"].type(Tensor))
        output = generator(input)
        img_sample = torch.cat((input.data, output.data, target.data), 2)
        save_image(img_sample, "validImages/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

    ##### Monitor the temperature of the device to avoid overheating #####
    def avg(value_list):
        num = 0
        length = len(value_list)
        for val in value_list:
            num += val
        return num / length

    def temp_monitor():
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()
        cpu_temps = []
        gpu_temp = 0
        for sensor in sensors:
            if sensor.SensorType == u'Temperature' and not 'GPU' in sensor.Name:
                cpu_temps += [float(sensor.Value)]
            elif sensor.SensorType == u'Temperature' and 'GPU' in sensor.Name:
                gpu_temp = sensor.Value

        # print("Avg CPU: {}".format(avg(cpu_temps)))
        # print("GPU: {}".format(gpu_temp))
        if avg(cpu_temps) > opt.device_temperature or gpu_temp > opt.device_temperature:
            print("Avg CPU: {}".format(avg(cpu_temps)))
            print("GPU: {}".format(gpu_temp))
            print("sleeping 30s")
            time.sleep(30)
        return

    ########################################
    #####           Training           #####
    ########################################

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            ##### Model Inputs: input pics and target pics #####
            input = Variable(batch["B"].type(Tensor))
            target = Variable(batch["A"].type(Tensor))

            ##### Ground truth for discriminator (for GAN loss calculation) #####
            valid = Variable(Tensor(np.ones((input.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            ##### Generator Losses #####
            fake_B = generator(input)
            pred_fake = discriminator(fake_B, input)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, target)
            loss_VGG = criterion_VGG(fake_B, target)

            ##### Total Generator Loss #####
            loss_G = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel + lambda_vgg * loss_VGG

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            ##### Discriminator Losses #####
            # Real loss
            pred_real = discriminator(target, input)
            loss_real = criterion_GAN(pred_real, valid)
            # Fake loss
            pred_fake = discriminator(fake_B.detach(), input)
            loss_fake = criterion_GAN(pred_fake, fake)

            ##### Total Discriminator Loss #####
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            ##### Approximate finishing time #####
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            ##### Print log #####
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, VGG: %f] EstimateTime: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    loss_VGG.item(),
                    time_left,
                )
            )

            ##### Save samples at intervals #####
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)
                temp_monitor()

        ##### Save model checkpoints #####
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(),
                       "saved_models/%s/%s_generator_%d.pth" % (opt.dataset_name, opt.model_name, epoch))
            torch.save(discriminator.state_dict(),
                       "saved_models/%s/%s_discriminator_%d.pth" % (opt.dataset_name, opt.model_name, epoch))

        ##### Optional logs in each epoch (by wandb) #####
        PSNR, SSIM = output_metrics()
        wandb.log({
            "loss": loss_G.item(),
            "%s_PSNR_%s" % (opt.dataset_name, "Generator"): PSNR,
            "%s_SSIM_%s" % (opt.dataset_name, "Generator"): SSIM,
        })
        wandb.watch(generator)


if __name__ == '__main__':
    ##### arguments settings #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=202, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="Dataset2", help="name of the dataset")
    parser.add_argument("--model_name", type=str, default="model_RCRN", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500,
                        help="interval between sampling of validImages from training")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument("--device_temperature", type=int, default=75,
                        help="Set maximum temperature of CPU and GPU for device safety")
    opt = parser.parse_args()

    ##### wandb initializing (Optional) #####
    wandb.login()
    wandb.init(project="LowLevel_demo", entity="daqian", name="%s_1" % opt.model_name)
    # wandb.config.name = opt.model_name
    # wandb.config = {
    #     "name": opt.model_name,
    #     "learning_rate": 0.001,
    #     "epochs": 100,
    #     "batch_size": 128,
    # }

    ##### Run training process #####
    TrainModel(opt)