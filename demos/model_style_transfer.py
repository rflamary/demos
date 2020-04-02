"""
MIT License

Copyright (c) 2020 Rémi Flamary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# This file in a merging of several files from 
# https://github.com/irasin/Pytorch_WCT

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x


vgg_decoder_relu5_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, 3)
    )


class Decoder(nn.Module):
    def __init__(self, level, pretrained_path=None):
        super().__init__()
        if level == 1:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-2:]))
        elif level == 2:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-9:]))
        elif level == 3:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-16:]))
        elif level == 4:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-29:]))
        elif level == 5:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())))
        else:
            raise ValueError('level should be between 1~5')

        if pretrained_path is not None:
            self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def forward(self, x):
        return self.net(x)


normalised_vgg_relu5_1 = nn.Sequential(
	nn.Conv2d(3, 3, 1),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3, 64, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 64, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 128, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 128, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU()
	)


class NormalisedVGG(nn.Module):
	"""
	VGG reluX_1(X = 1, 2, 3, 4, 5) can be obtained by slicing the follow vgg5_1 model.

	Sequential(
	(0): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
	(1): ReflectionPad2d((1, 1, 1, 1))
	(2): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
	(3): ReLU() # relu1_1
	(4): ReflectionPad2d((1, 1, 1, 1))
	(5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
	(6): ReLU()
	(7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(8): ReflectionPad2d((1, 1, 1, 1))
	(9): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
	(10): ReLU() # relu2_1
	(11): ReflectionPad2d((1, 1, 1, 1))
	(12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
	(13): ReLU()
	(14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(15): ReflectionPad2d((1, 1, 1, 1))
	(16): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
	(17): ReLU() # relu3_1
	(18): ReflectionPad2d((1, 1, 1, 1))
	(19): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
	(20): ReLU()
	(21): ReflectionPad2d((1, 1, 1, 1))
	(22): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
	(23): ReLU()
	(24): ReflectionPad2d((1, 1, 1, 1))
	(25): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
	(26): ReLU()
	(27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(28): ReflectionPad2d((1, 1, 1, 1))
	(29): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
	(30): ReLU()# relu4_1
	(31): ReflectionPad2d((1, 1, 1, 1))
	(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(33): ReLU()
	(34): ReflectionPad2d((1, 1, 1, 1))
	(35): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(36): ReLU()
	(37): ReflectionPad2d((1, 1, 1, 1))
	(38): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(39): ReLU()
	(40): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(41): ReflectionPad2d((1, 1, 1, 1))
	(42): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(43): ReLU() # relu5_1
	)
	"""
	def __init__(self, pretrained_path='vgg_normalised_conv5_1.pth'):
		super().__init__()
		self.net = normalised_vgg_relu5_1
		if pretrained_path is not None:
			self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

	def forward(self, x, target):
		if target == 'relu1_1':
			return self.net[:4](x)
		elif target == 'relu2_1':
			return self.net[:11](x)
		elif target == 'relu3_1':
			return self.net[:18](x)
		elif target == 'relu4_1':
			return self.net[:31](x)
		elif target == 'relu5_1':
			return self.net(x)
		else:
			raise ValueError(f'target should be in ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"] but not {target}')


def whiten_and_color(content_feature, style_feature, alpha=1):
    """
    A WCT function can be used directly between encoder and decoder
    """
    cf = content_feature.squeeze(0)#.double()
    c, ch, cw = cf.shape
    cf = cf.reshape(c, -1)
    c_mean = torch.mean(cf, 1, keepdim=True)
    cf = cf - c_mean
    c_cov = torch.mm(cf, cf.t()).div(ch*cw - 1)
    c_u, c_e, c_v = torch.svd(c_cov)

    # if necessary, use k-th largest eig-value
    k_c = c
    for i in range(c):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = c_e[:k_c].pow(-0.5)

    w_step1 = torch.mm(c_v[:, :k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, :k_c].t()))
    whitened = torch.mm(w_step2, cf)

    sf = style_feature.squeeze(0)#.double()
    c, sh, sw = sf.shape
    sf = sf.reshape(c, -1)
    s_mean = torch.mean(sf, 1, keepdim=True)
    sf = sf - s_mean
    s_cov = torch.mm(sf, sf.t()).div(sh*sw -1)
    s_u, s_e, s_v = torch.svd(s_cov)

    # if necessary, use k-th largest eig-value
    k_s = c
    for i in range(c):
        if s_e[i] < 0.00001:
            k_s = i
            break
    s_d = s_e[:k_s].pow(0.5)
    c_step1 = torch.mm(s_v[:, :k_s], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, :k_s].t())
    colored = torch.mm(c_step2, whitened) + s_mean

    colored_feature = colored.reshape(c, ch, cw).unsqueeze(0).float()

    colored_feature = alpha * colored_feature + (1 - alpha) * content_feature
    return colored_feature

class SingleLevelAE(nn.Module):
    def __init__(self, level, pretrained_path_dir='model_state'):
        super().__init__()
        self.level = level
        self.encoder = NormalisedVGG(f'{pretrained_path_dir}/vgg_normalised_conv5_1.pth')
        self.decoder = Decoder(level, f'{pretrained_path_dir}/decoder_relu{level}_1.pth')

    def forward(self, content_image, style_image, alpha):
        content_feature = self.encoder(content_image, f'relu{self.level}_1')
        style_feature = self.encoder(style_image, f'relu{self.level}_1')
        res = whiten_and_color(content_feature, style_feature, alpha)
        res = self.decoder(res)
        return res


class MultiLevelAE(nn.Module):
    def __init__(self, pretrained_path_dir='model_state'):
        super().__init__()
        self.encoder = NormalisedVGG(f'{pretrained_path_dir}/vgg_normalised_conv5_1.pth')
        self.decoder1 = Decoder(1, f'{pretrained_path_dir}/decoder_relu1_1.pth')
        self.decoder2 = Decoder(2, f'{pretrained_path_dir}/decoder_relu2_1.pth')
        self.decoder3 = Decoder(3, f'{pretrained_path_dir}/decoder_relu3_1.pth')
        self.decoder4 = Decoder(4, f'{pretrained_path_dir}/decoder_relu4_1.pth')
        self.decoder5 = Decoder(5, f'{pretrained_path_dir}/decoder_relu5_1.pth')

    def transform_level(self, content_image, style_image, alpha, level):
        content_feature = self.encoder(content_image, f'relu{level}_1')
        style_feature = self.encoder(style_image, f'relu{level}_1')
        res = whiten_and_color(content_feature, style_feature, alpha)
        return getattr(self, f'decoder{level}')(res)

    def forward(self, content_image, style_image, alpha=1):
        r5 = self.transform_level(content_image, style_image, alpha, 5)
        r4 = self.transform_level(r5, style_image, alpha, 4)
        r3 = self.transform_level(r4, style_image, alpha, 3)
        r2 = self.transform_level(r3, style_image, alpha, 2)
        r1 = self.transform_level(r2, style_image, alpha, 1)

        return r1

