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
from datetime import datetime
import os
import urllib
import ot

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

folder = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
style_path = dir_path+'/../data/styles/'

if not os.path.exists('out'):
    os.mkdir('out')

idimg = 0

fname = "out/{}/{}_{}.jpg"

lst_map = ['Linear OT','Exact OT', 'Sinkhorn OT']
id_map = 0

n_keep = 200

def transfer(c, s, alpha=1):
    nxs = c.shape[0]*c.shape[1]
    nxt = s.shape[0]*s.shape[1]
    xs = c.copy().reshape((nxs,3))*1.0
    xt = s.copy().reshape((nxt,3))*1.0
    if id_map==0:
        adapt = ot.da.LinearTransport()
        adapt.fit(xs, Xt=xt)
    elif id_map==1:
        adapt = ot.da.EMDTransport()
        xs2 = xs[np.random.permutation(xs.shape[0])[:n_keep],:]
        xt2 = xt[np.random.permutation(xt.shape[0])[:n_keep],:]
        adapt.fit(xs2, Xt=xt2)
    elif id_map==2:
        adapt = ot.da.SinkhornTransport(reg_e=100)
        xs2 = xs[np.random.permutation(xs.shape[0])[:n_keep],:]
        xt2 = xt[np.random.permutation(xt.shape[0])[:n_keep],:]
        adapt.fit(xs2, Xt=xt2)
    return adapt.transform(xs).reshape(c.shape)


cam = os.getenv("CAMERA")
if cam is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(int(cam))

lst_style = ['antimonocromatismo.jpg',
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
             'nikopol.jpg', ]

lst_style0 = lst_style

lst_name = [f.split('.')[0] for f in lst_style]

lst_style = [np.array(Image.open(style_path+f).convert('RGB'))
             for f in lst_style]

col = [255, 1, 1]
max_iters = 1
alpha = 0.8
id_style = 0

from_RGB = [2, 1, 0]


pause = False

ret, frame0 = cap.read()
frame0 = np.asfortranarray(np.array(frame0)/255)
frame_webcam = frame0
frame_style = frame0*0


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame2 = np.array(frame)
    frame2 = frame2[:, ::-1, :]

    if not pause:
        frame_webcam = frame2

    # Display the images
    cv2.imshow('Webcam', frame_webcam)
    cv2.imshow('Transferred image ({})'.format(lst_map[id_map]), frame_style[:, :, from_RGB])

    cv2.namedWindow('Target Color', cv2.WINDOW_NORMAL)
    cv2.imshow('Target Color', lst_style[id_style][:, :, from_RGB])

    # handle inputs
    key = cv2.waitKey(1)
    if (key & 0xFF) in [ord('q')]:
        break
    if (key & 0xFF) in [ord('s')]:
        id_style = (id_style+1) % len(lst_name)
    if (key & 0xFF) in [ord('w')]:
        if not os.path.exists('out/{}'.format(folder)):
            os.mkdir('out/{}'.format(folder))
        cv2.imwrite(fname.format(folder, idimg, '0'), frame_webcam)
        cv2.imwrite(fname.format(folder, idimg,
                    lst_name[id_style]), frame_style[:, :, from_RGB]*255)
        print("Images saved")
    if (key & 0xFF) in [ord('q')]:
        break
    if (key & 0xFF) in [ord('r')]:
        if not os.path.exists('out/{}'.format(folder)):
            os.mkdir('out/{}'.format(folder))
        cv2.imwrite(fname.format(folder, idimg, '0'), frame_webcam)
        for i in range(len(lst_style)):
            temp = np.array(transfer(frame_webcam, lst_style[i], alpha=alpha))
        # print(temp)
            for k in range(3):
                frame_style[:, :, k] = temp[k, :, :]/255
            cv2.imwrite(fname.format(folder, idimg,
                        lst_name[i]), frame_style[:, :, from_RGB]*255)
            print('Applied style from file {}'.format(lst_style0[i]))
            cv2.imshow('Transferred image ({})'.format(
                lst_style0[i]), frame_style[:, :, from_RGB])

    if (key & 0xFF) in [ord('p')]:
        pause = not pause
        if pause:
            idimg += 1
    if (key & 0xFF) in [ord('A')]:
        alpha = min(1, alpha+0.1)
        print('alpha={}'.format(alpha))
    if (key & 0xFF) in [ord('a')]:
        alpha = max(0, alpha-0.1)
        print('alpha={}'.format(alpha))
    if (key & 0xFF) in [ord('m')]:
        id_map = (id_map+1) % len(lst_map)
        temp = np.array(
            transfer(frame_webcam, lst_style[id_style], alpha=alpha))
        # print(temp)
        for i in range(3):
            frame_style[:, :, i] = temp[ :, :, i]/255
        print('Applied style from file {}'.format(lst_style0[id_style]))        
    if (key & 0xFF) in [ord(' ')]:
        pause = True
        temp = np.array(
            transfer(frame_webcam, lst_style[id_style], alpha=alpha))
        # print(temp)
        for i in range(3):
            frame_style[:, :, i] = temp[ :, :, i]/255
        print('Applied style from file {}'.format(lst_style0[id_style]))


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
