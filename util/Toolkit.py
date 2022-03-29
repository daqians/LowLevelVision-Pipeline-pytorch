import torch
from models.models3_densenet_dilated_parallel import GeneratorUNet






##### Save model weight into the old format torch version<1.6 #####
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GeneratorUNet().to(device)
model.load_state_dict(torch.load("saved_models/Oracles/generator_200.pth",map_location=device))

""" The 1.6 release of PyTorch switched torch.save to use a new zipfile-based file format.
torch.load still retains the ability to load files in the old format.
If for any reason you want torch.save to use the old format, pass the kwarg _use_new_zipfile_serialization=False. """
torch.save(model.state_dict(), "../saved_models/Oracles/generator_200k.pth", _use_new_zipfile_serialization=False)
