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
import scipy.ndimage
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model_style_transfer import MultiLevelAE
from datetime import datetime
import os
import urllib

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

style_path=dir_path+'/../data/styles/'
models_path=dir_path+'/../data/models'
url_models='https://remi.flamary.com/download/models/'

folder=datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

trans = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('out'):
    os.mkdir('out')


lst_model_files=[ "decoder_relu1_1.pth",
                  "decoder_relu2_1.pth",
                  "decoder_relu3_1.pth",
                  "decoder_relu4_1.pth",
                  "decoder_relu5_1.pth",
                  "vgg_normalised_conv5_1.pth"]

# test if models already downloaded
for m in lst_model_files:
    if not os.path.exists(models_path+'/'+m):
        print('Downloading model file : {}'.format(m))
        urllib.request.urlretrieve(url_models+m,models_path+'/'+m)


idimg=0

fname="out/{}/{}_{}.jpg"

if torch.cuda.is_available():
    device = torch.device(f'cuda')
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'


model = MultiLevelAE(models_path)
model = model.to(device)
print("Model loaded")

def transfer(c,s,alpha=1):
    c=c[:,:,::-1].copy()
    c_tensor = trans(c.astype(np.float32)).unsqueeze(0).to(device)
    s_tensor = trans(s.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(c_tensor, s_tensor, alpha)
    return out.numpy()[0,:,:,:]

cam=os.getenv("CAMERA")
if cam is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(int(cam))

lst_style=['antimonocromatismo.jpg',
            'sketch.png',
            'hosi.jpg',
            'asheville.jpg',
            'brushstrokes.jpg',
            'contrast_of_forms.jpg',
            'en_campo_gris.jpg',
            'la_muse.jpg',
            'mondrian.jpg',
            'picasso_seated_nude_hr.jpg',
            'picasso_self_portrait.jpg',
            'scene_de_rue.jpg',
            'trial.jpg',
            'woman_in_peasant_dress_cropped.jpg',
            'woman_with_hat_matisse.jpg',
            'afremov.jpg',
            'bubbles.jpg',
            'in2.jpg',
            'baiser.jpg',
            'zombie.jpg',
            'nikopol.jpg',
            'miro.jpg',
            'monet1.jpg',
            'monet2.jpg',
            'nikopol.jpg',]

lst_style0=lst_style

lst_name=[f.split('.')[0] for f in lst_style]

lst_style=[np.array(Image.open(style_path+f).convert('RGB')) for f in lst_style]

col=[255,1,1]
max_iters=1
alpha=0.8
id_style=0

from_RGB=[2,1,0]


pause=False

ret, frame0 = cap.read()
frame0=np.asfortranarray(np.array(frame0)/255)
frame_webcam=frame0
frame_style = frame0*0


resize_shape = os.getenv('RESIZE', '1280x720').split('x')
assert len(resize_shape) == 2, "Error: invalid resize parameter: {}".format(resize_shape)
resize_shape = tuple([int(i) for i in resize_shape[::-1]] + [3])[::-1]

# When using standard resolution of 1280x720 we don't need to resize the output because
# the model will predict frames of the same size
resize = False
if frame_style.shape != resize_shape[::-1]:
    # For other resolution like for example smaller ones, the output is resized
    resize = True
    print("Warning, resizing your webcame image from {}x{}x3 to {}x{}x3!".format(frame_style.shape[1], \
            frame_style.shape[0], resize_shape[1], resize_shape[0]))
    frame_style = cv2.resize(frame_style, resize_shape[1:])
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
        
    frame2=np.array(frame)
    frame2=frame2[:,::-1,:]

    if not pause:
        frame_webcam=frame2

    # Display the images
    cv2.imshow('Webcam',frame_webcam)
    cv2.imshow('Transferred image',frame_style[:,:,from_RGB])
    cv2.namedWindow('Target Style', cv2.WINDOW_NORMAL)
    cv2.imshow('Target Style',lst_style[id_style][:,:,from_RGB])

    # handle inputs
    key=cv2.waitKey(1)
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        id_style=(id_style+1)%len(lst_name)
    if (key & 0xFF) in [ ord('w')]:
        if not os.path.exists('out/{}'.format(folder)):
            os.mkdir('out/{}'.format(folder))
        cv2.imwrite(fname.format(folder,idimg,'0'),frame_webcam)
        cv2.imwrite(fname.format(folder,idimg,lst_name[id_style]),frame_style[:,:,from_RGB]*255)
        print("Images saved")
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('r')]:
        if not os.path.exists('out/{}'.format(folder)):
            os.mkdir('out/{}'.format(folder))

        if resize:
            frame_webcam = cv2.resize(frame_webcam, resize_shape[1:])

        cv2.imwrite(fname.format(folder,idimg,'0'),frame_webcam)
        for i in range(len(lst_style)):
            temp=np.array(transfer(frame_webcam,lst_style[i],alpha=alpha))
            for k in range(3):
                frame_style[:,:,k]=temp[k,:,:]/255
            cv2.imwrite(fname.format(folder,idimg,lst_name[i]),frame_style[:,:,from_RGB]*255)
            print('Applied style from file {}'.format(lst_style0[i]))
            cv2.imshow('Transferred image ({})'.format(lst_style0[i]),frame_style[:,:,from_RGB])

    if (key & 0xFF) in [ ord('p')]:
        pause = not pause
        if pause:
            idimg+=1
    if (key & 0xFF) in [ ord('A')]:
        alpha=min(1,alpha+0.1)
        print('alpha={}'.format(alpha))    
    if (key & 0xFF) in [ ord('a')]:
        alpha=max(0,alpha-0.1)  
        print('alpha={}'.format(alpha))      
    if (key & 0xFF) in [ ord(' ')]:
        pause=True

        if resize:
            frame_webcam = cv2.resize(frame_webcam, resize_shape[1:])

        temp=transfer(frame_webcam,lst_style[id_style],alpha=alpha)
        if resize:
            temp = np.transpose(temp, (1,2,0))
            temp = cv2.resize(temp, resize_shape[1:])
            temp = np.transpose(temp, (2,0,1))
        for i in range(3):
            frame_style[:,:,i]=temp[i,:,:]/255
        print('Applied style from file {}'.format(lst_style0[id_style]))


    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
