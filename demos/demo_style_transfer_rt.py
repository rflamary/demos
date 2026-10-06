#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from model_style_transfer_rt import AdaINStyleTransfer
from datetime import datetime
import os
import urllib.request
import warnings

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-r',
    '--downscale_ratio',
    type=float,
    default=1.0,
    help='Downscale ratio for internal processing'
)
args = parser.parse_args()
DOWNSCALE_RATIO = args.downscale_ratio
print('Internal processing downscale ratio: {}'.format(DOWNSCALE_RATIO))


import os
dir_path = os.path.dirname(os.path.realpath(__file__))

style_path = dir_path+'/../data/styles/'
models_path = dir_path+'/../data/models'
url_models = 'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/'

folder = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

trans = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists(models_path):
    os.makedirs(models_path)


lst_model_files = ["decoder.pth",
                   "vgg_normalised.pth"]

# test if models already downloaded
for m in lst_model_files:
    if not os.path.exists(models_path+'/'+m):
        print('Downloading model file : {}'.format(m))
        urllib.request.urlretrieve(url_models+m, models_path+'/'+m)


idimg = 0

fname = "out/{}/{}_{}.jpg"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CONTENT_STD_BLEND = 0.45
STYLE_MAX_SIZE = 512
ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_AMP = True

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('# MPS (Apple Metal) is available. Using Mac GPU.')
    warnings.filterwarnings('ignore')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
else:
    device = 'cpu'
    print('# No GPU found. Using CPU.')


model = AdaINStyleTransfer(
    decoder_path=models_path+'/decoder.pth',
    vgg_path=models_path+'/vgg_normalised.pth')
model = model.to(device).eval()

if ENABLE_TORCH_COMPILE and torch.cuda.is_available() and hasattr(torch, 'compile'):
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print('# torch.compile enabled.')
    except Exception as exc:
        print('# torch.compile unavailable/failed, continuing normally: {}'.format(exc))

print("Model loaded")


def transfer(c, style_id, alpha=1, resize=True):
    c = c[:, :, ::-1].copy()
    if resize:
        h, w = c.shape[:2]
        c = cv2.resize(c,
                       (max(1, int(w/DOWNSCALE_RATIO)),
                        max(1, int(h/DOWNSCALE_RATIO))),
                       interpolation=cv2.INTER_AREA)

    c_tensor = trans(c.astype(np.float32) / 255.0).unsqueeze(0).to(device)

    with torch.inference_mode():
        if ENABLE_CUDA_AMP and torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model.forward_with_cached_style(
                    c_tensor, style_features[style_id], alpha, CONTENT_STD_BLEND)
        else:
            out = model.forward_with_cached_style(
                c_tensor, style_features[style_id], alpha, CONTENT_STD_BLEND)

    out = torch.clamp(out.float(), 0.0, 1.0)
    return out[0, :, :, :].detach().cpu().numpy()



cam = os.getenv("CAMERA")
if cam is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(int(cam))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


lst_style = ['antimonocromatismo.jpg',
             'hosi.jpg',
             'contrast_of_forms.jpg',
             'la_muse.jpg',
             'mondrian.jpg',
             'picasso_seated_nude_hr.jpg',
             'woman_with_hat_matisse.jpg',
             'afremov.jpg',
             'zombie.jpg',
             'monet1.jpg',
             'dragibus.png']

lst_style0 = lst_style

lst_name = [f.split('.')[0] for f in lst_style]


def resize_style(img, max_size=STYLE_MAX_SIZE):
    w, h = img.size
    scale = max_size/max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w*scale)),
                          max(1, int(h*scale))), Image.BICUBIC)
    return img


lst_style = [np.array(resize_style(Image.open(style_path+f).convert('RGB')))
             for f in lst_style]

style_features = []
with torch.inference_mode():
    for i in range(len(lst_style)):
        s_tensor = trans(lst_style[i]).unsqueeze(0).to(device)
        style_features.append(model.encode_style(s_tensor))
        print('Cached style: {}'.format(lst_style0[i]))


col = [255, 1, 1]
max_iters = 1
alpha = 0.8
id_style = 0

from_RGB = [2, 1, 0]

pause = False
resize = True
continuous = False


ret, frame0 = cap.read()
if not ret:
    cap.release()
    raise RuntimeError('Could not read from camera.')

frame_webcam = np.array(frame0)
frame_style = np.zeros_like(frame_webcam, dtype=np.float32)

cv2.namedWindow('Webcam', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Transferred image', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Target Style', cv2.WINDOW_NORMAL)
cv2.setWindowTitle('Webcam', 'Webcam - Rescaling: ON')

print("""
Controls:
  q       quit
  space   apply current style to current frame
  v       toggle continuous live transfer
  s       next style
  z       toggle internal resize
  p       pause/unpause webcam preview
  a/A     decrease/increase alpha
  c/C     decrease/increase content-contrast preservation
  w       save current webcam + stylized image
  r       apply all styles to current frame
""")


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        continue

    frame2 = np.array(frame)

    if not pause:
        frame_webcam = frame2

    # Display the images
    cv2.imshow('Webcam', frame_webcam)
    h, w = frame_webcam.shape[:2]
    display_style = cv2.resize(
        frame_style[:, :, from_RGB],
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )
    cv2.imshow('Transferred image', display_style)

    cv2.imshow('Target Style', lst_style[id_style][:, :, from_RGB])

    # handle inputs
    key = cv2.waitKey(1)
    if (key & 0xFF) in [ord('q')]:
        break
    if (key & 0xFF) in [ord('s')]:
        id_style = (id_style+1) % len(lst_name)
        print('Selected style: {}'.format(lst_style0[id_style]))
    if (key & 0xFF) in [ord('w')]:
        if not os.path.exists('out/{}'.format(folder)):
            os.mkdir('out/{}'.format(folder))
        cv2.imwrite(fname.format(folder, idimg, '0'), frame_webcam)
        cv2.imwrite(fname.format(folder, idimg,
                    lst_name[id_style]), (frame_style[:, :, from_RGB]*255).astype(np.uint8))
        print("Images saved")
    if (key & 0xFF) in [ord('q')]:
        break
    if (key & 0xFF) in [ord('r')]:
        if not os.path.exists('out/{}'.format(folder)):
            os.mkdir('out/{}'.format(folder))
        cv2.imwrite(fname.format(folder, idimg, '0'), frame_webcam)
        for i in range(len(lst_style)):
            temp = np.array(transfer(frame_webcam, i,
                                     alpha=alpha, resize=resize))
            frame_style = temp.transpose(1, 2, 0)
            cv2.imwrite(fname.format(folder, idimg,
                        lst_name[i]), (frame_style[:, :, from_RGB]*255).astype(np.uint8))
            print('Applied style from file {}'.format(lst_style0[i]))
            cv2.imshow('Transferred image ({})'.format(
                lst_style0[i]), frame_style[:, :, from_RGB])

    if (key & 0xFF) in [ord('p')]:
        pause = not pause
        if pause:
            idimg += 1
        print('Pause: {}'.format(pause))
    if (key & 0xFF) in [ord('A')]:
        alpha = min(1, alpha+0.1)
        print('alpha={}'.format(alpha))
    if (key & 0xFF) in [ord('a')]:
        alpha = max(0, alpha-0.1)
        print('alpha={}'.format(alpha))
    if (key & 0xFF) in [ord('C')]:
        CONTENT_STD_BLEND = min(0.95, CONTENT_STD_BLEND+0.05)
        print('content_std_blend={}'.format(CONTENT_STD_BLEND))
    if (key & 0xFF) in [ord('c')]:
        CONTENT_STD_BLEND = max(0, CONTENT_STD_BLEND-0.05)
        print('content_std_blend={}'.format(CONTENT_STD_BLEND))
    if (key & 0xFF) in [ord('z')]:
        resize = not resize
        state = 'ON' if resize else 'OFF'
        cv2.setWindowTitle('Webcam', 'Webcam - Rescaling: {}'.format(state))
        h, w = frame_webcam.shape[:2]
        print('Internal processing resize toggled: {} '
              '(NN running at {}x{} when ON)'.format(
                  state, int(w/DOWNSCALE_RATIO), int(h/DOWNSCALE_RATIO)))
    if (key & 0xFF) in [ord('v')]:
        continuous = not continuous
        state = 'ON (Live Video)' if continuous else 'OFF (Manual Spacebar)'
        print('Continuous mode toggled: {}'.format(state))
    if (key & 0xFF) in [ord(' ')]:
        pause = True
        temp = np.array(
            transfer(frame_webcam, id_style, alpha=alpha, resize=resize))
        frame_style = temp.transpose(1, 2, 0)
        print('Applied style from file {}'.format(lst_style0[id_style]))

    if continuous and not pause:
        temp = np.array(
            transfer(frame_webcam, id_style, alpha=alpha, resize=resize))
        frame_style = temp.transpose(1, 2, 0)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()