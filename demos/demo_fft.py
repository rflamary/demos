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
import prox_tv as ptv
import os
import scipy as sp
import scipy.signal


cam=os.getenv("CAMERA")
if cam is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(int(cam))

reg=2e-1
alpha=0.2
pred=0
idscreen=0
loop=True
do_fft=True
half=False


tau_noise=0.002

ret, frame0 = cap.read()

n1=frame0.shape[0]
n2=frame0.shape[1]

win=False

w1=sp.signal.get_window('parzen',n1)**.2
w2=sp.signal.get_window('parzen',n2)**.2
W=w1[:,None]*w2[None,:]

scale = 0

scales= ["Linear",'Sqrt','Log']


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    
    frame2=np.array(frame)/255
    frame2=frame2[:,::-1,:]
    #frame2=frame2[::2,::2,:]

    if win:
        frame2=frame2*W[:,:,None]
    
    F=np.zeros_like(frame2)
    for i in range(3):
        F[:,:,i]=np.abs(np.fft.fftshift(np.fft.fft2(frame2[:,:,i]),axes=(0,1)))
    Fl=np.log(F+1e-2)
    Fl/=Fl.max()

    F =F/F.max()*100

    Fs = np.sqrt(F)
    Fs = Fs/Fs.max()
    

    
    #frame2=(frame2+frame0)/2
    
    frame0=frame2
    
    #frame2-=frame2.min()
    #frame2/=frame2.max()
             
    # Display the resulting frame
    cv2.imshow('Webcam',frame2)

    if do_fft:
        cv2.imshow('FFT webcam Log scale',Fl)
        cv2.imshow('FFT webcam Linear scale',F)
        #cv2.imshow('FFT webcam Sqrt scale',F)

    key=cv2.waitKey(1)
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        #cv2.imwrite("screen_{}.png".format(idscreen),frame2*255)
        #idscreen+=1

        scale = 0
    if (key & 0xFF) in [ ord(' ')]:
        do_fft=not do_fft
    if (key & 0xFF) in [ ord('w')]:
        win=not win       
    if (key & 0xFF) in [ ord('h')]:
        if not half:
            do_tv=True
            half=True
        else:
            half=False
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
