import torch
from model_style_transfer import MultiLevelAE
from torchvision import transforms 
from PIL import Image

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
models_path = dir_path + '/../data/models'
url_models = 'https://remi.flamary.com/download/models/'
trans = transforms.Compose([transforms.ToTensor()])

lst_model_files=[ "decoder_relu1_1.pth",
                  "decoder_relu2_1.pth",
                  "decoder_relu3_1.pth",
                  "decoder_relu4_1.pth",
                  "decoder_relu5_1.pth",
                  "vgg_normalised_conv5_1.pth"]
trans = transforms.Compose([transforms.ToTensor()])

# test if models already downloaded
for m in lst_model_files:
    if not os.path.exists(models_path+'/'+m):
        print('Downloading model file : {}'.format(m))
        urllib.request.urlretrieve(url_models+m,models_path+'/'+m)


if torch.cuda.is_available():
    device = torch.device(f'cuda')
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'


model = MultiLevelAE(models_path)
model = model.to(device)
encoder = model.encoder
print("Model loaded")


x = '../data/styles/afremov.jpg'
x = trans(Image.open(x).convert('RGB'))
features5 = encoder(x, f'relu5_1')
print(features5.shape)
features4 = encoder(x, f'relu4_1')
print(features5.shape)
features3 = encoder(x, f'relu3_1')
print(features5.shape)
features2 = encoder(x, f'relu2_1')
print(features5.shape)
features1 = encoder(x, f'relu1_1')
print(features5.shape)
