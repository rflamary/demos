#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:44:57 2016

@author: rflamary
"""

import numpy as np
import pylab as pl
import numpy as np
import cv2
import pylab as pl
import json
cap = cv2.VideoCapture(0)


import PIL.Image
import keras
from keras.preprocessing import image

dico_french_file = '../data/imagenet_class_french2.json'
dico_french = json.load(open(dico_french_file))

model = 'VGG16'
model = 'resnet50'
#model = 'convnextsmall'

if model=='resnet50':
    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    model = ResNet50(weights='imagenet')
elif model=='convnextsmall':
    from keras.applications import ConvNeXtSmall
    model = ConvNeXtSmall(weights='imagenet')
elif model=='VGG16':
    from keras.applications import VGG16
    from keras.applications.vgg16 import preprocess_input, decode_predictions
    model=keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)


alpha=0.95
pred=0
idscreen=0
loop=False

col=[0,0,00]

loop_webcam=True
use_webcam=True
classif=False
noise_level=0
nb_noise_levels=10

ret, frame = cap.read()
size = frame.shape
print("Size of the webcam image: {}".format(size))

def get_noise():
    sigma = 5
    noise = np.zeros((nb_noise_levels, size[0], size[1], 3), dtype=np.float32)
    for i in range(1,nb_noise_levels):
        noise[i, :, :, :] = sigma*i*i*np.random.normal(0, sigma, (size[0], size[1], 3))+noise[i-1, :, :, :]
    return noise

def add_noise(frame, noise):
    # convert frame to numpy array if it is not already
    frame2 = np.array(frame, dtype=np.float32)
    # add noise to the frame
    noisy_frame = frame2 + noise[noise_level, :, :, :]
    # clip the values to be in the range [0, 255]
    noisy_frame = np.clip(noisy_frame, 0, 255)
    framen = frame.copy()
    for i in range(3):
        framen[:, :, i] = noisy_frame[:, :, i]

    
    return framen

noise = get_noise()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if loop_webcam:
        frame0 = frame.copy()
        framen = add_noise(frame0, noise)
        screen = framen.copy()
    else:
        framen = add_noise(frame0, noise)
        screen=framen.copy()

    key=cv2.waitKey(1)
    # if not key == -1:
    #     print(key)
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        cv2.imwrite("screen_{}.png".format(idscreen),frame)
        idscreen+=1
    if (key & 0xFF) in [ ord(' ')]:
        loop_webcam=not loop_webcam
        noise = get_noise()
    if (key & 0xFF) in [ ord('c')]:
        classif=not classif
    if (key & 0xFF) in [ ord('n')]:
        noise = get_noise()
    if key == 83 or key == 3:  # Right arrow key
        noise_level = min( nb_noise_levels-1, noise_level + 1)
            
        print("Noise level: {}".format(noise_level))
    if key == 81 or key == 2:  # Left arrow key
        noise_level = max(0, noise_level - 1)
        print("Noise level: {}".format(noise_level))
         
    if classif:

        img=np.array(cv2.resize(cv2.cvtColor(framen, cv2.COLOR_BGR2RGB),(224,224)))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        if loop:
            pred = alpha*pred+(1-alpha)*model.predict(x)
        else:
            pred = model.predict(x)

        txt=decode_predictions(pred,5)
        #print txt
        for i,p in enumerate(txt[0]):
            text = "{:.3f}: {}".format(float(p[2]), dico_french[p[1]])
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
            x, y = 0, 28 * i + 30
            cv2.rectangle(screen, (x, y - text_height - 5), (x + text_width + 10, y + 5), (255, 255, 255), -1)
            cv2.putText(screen, text, (x + 5, y), cv2.FONT_HERSHEY_DUPLEX, 1, col, 1)
                        
    # Display the resulting frame
    cv2.imshow('Diffusion Demo',screen)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
