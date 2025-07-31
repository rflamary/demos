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

col=[0,255,00]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame2=frame


    key=cv2.waitKey(1)
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        cv2.imwrite("screen_{}.png".format(idscreen),frame)
        idscreen+=1
    if (key & 0xFF) in [ ord(' ')]:
        loop=not loop
        
    if loop:
        img=np.array(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(224,224)))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pred = alpha*pred+(1-alpha)*model.predict(x)

        txt=decode_predictions(pred,5)
        #print txt
        for i,p in enumerate(txt[0]):
            cv2.putText(frame,"{:.3f}: {}".format(float(p[2]),dico_french[p[1]]),(0,28*i+30),cv2.FONT_HERSHEY_DUPLEX,1,col,1)

            
    # Display the resulting frame
    cv2.imshow('frame',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
