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
filt=False
idfilt=0

s=1
cut=0.5


font                   = cv2.FONT_HERSHEY_PLAIN
bottomLeftCornerOfText = (10,30)
fontScale              = 2
fontColor              = (255,0,0)
lineType               = 2


ret, frame0 = cap.read()

n1=frame0.shape[0]
n2=frame0.shape[1]

win=False

w1=sp.signal.get_window('parzen',n1)**.2
w2=sp.signal.get_window('parzen',n2)**.2
W=w1[:,None]*w2[None,:]

F1,F2=np.meshgrid(np.fft.fftfreq(n2),np.fft.fftfreq(n1))
R=np.sqrt(F1**2+F2**2)


def filter_img(x,idfilt):

    idfilt= idfilt%13

    if idfilt==0:
        apply=lambda x: x
        txt= 'No filter'
    if idfilt==1:
        def apply(x):
            h=np.ones((2*s+1,2*s+1))/(2*s+1)**2
            return sp.signal.convolve(x,h,'same')
        txt= 'Average filter size={}'.format(s*2+1)        
    if idfilt==2:
        def apply(x):
            h=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])/4
            return np.maximum(np.minimum(sp.signal.convolve(x,h,'same'),1),0)
        txt= 'High pass'
    if idfilt==3:
        def apply(x):
            h=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])/4
            return abs(sp.signal.convolve(x,h,'same'))**.5
        txt= 'High pass Abs.'
    if idfilt==4:
        def apply(x):
            h=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])/3
            return np.maximum(np.minimum(sp.signal.convolve(x,h,'same'),1),0)
        txt= 'Prewit Horiz. Abs.'
    if idfilt==5:
        def apply(x):
            h=h=np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).T/3
            return np.maximum(np.minimum(sp.signal.convolve(x,h,'same'),1),0)
        txt= 'Prewit Vert. Abs.'
    if idfilt==6:
        def apply(x):
            h=h=np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).T/3
            return abs(sp.signal.convolve(x,h,'same'))**.5+abs(sp.signal.convolve(x,h.T,'same'))**.5
        txt= 'Prewit Both Abs.'       
    if idfilt==7:
        def apply(x):
            h=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/4
            return np.maximum(np.minimum(sp.signal.convolve(x,h,'same'),1),0)
        txt= 'Sobel Horiz. Abs.'
    if idfilt==8:
        def apply(x):
            h=h=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T/4
            return np.maximum(np.minimum(sp.signal.convolve(x,h,'same'),1),0)
        txt= 'Sobel Vert. Abs.'
    if idfilt==9:
        def apply(x):
            h=h=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T/4
            return abs(sp.signal.convolve(x,h,'same'))**.5+abs(sp.signal.convolve(x,h.T,'same'))**.5
        txt= 'Sobel Both Abs.'       
    if idfilt==10:
        def apply(x):
            return sp.signal.medfilt2d(x,2*s+1)
        txt= 'Median filter size={}'.format(s*2+1)    
    if idfilt==11:
        def apply(x):
            return np.fft.ifft2(np.fft.fft2(x)*(R<cut))
        txt= 'Ideal low-pass cutoff={}'.format(cut)
    if idfilt==12:
        def apply(x):
            minv=x.min()
            maxv=x.max()
            return np.clip(np.fft.ifft2(np.fft.fft2(x)*(R>cut)),minv,maxv)
        txt= 'Ideal high-pass cutoff={}'.format(cut)        
    xf=np.zeros_like(x)
    for i in range(3):
        xf[:,:,i]=apply(x[:,:,i])

    dt=5
    xf+=xf[dt:-dt,dt:-dt,:].min()
    xf/=xf[dt:-dt,dt:-dt,:].max()

    return xf, txt


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    
    frame2=np.array(frame)*1.0/255
    #frame2=frame2[:,::-1,:]
    #frame2=frame2[::2,::2,:]

    if win:
        frame2=frame2*W[:,:,None]

    if do_fft:
        frame2,txt=filter_img(frame2,idfilt)
    else:
        txt='No filter'

    
    
    F=np.zeros_like(frame2)
    for i in range(3):
        F[:,:,i]=np.abs(np.fft.fftshift(np.fft.fft2(frame2[:,:,i]),axes=(0,1)))
    F=np.log(F+1e-2)
    F/=F.max()
    

    
    #frame2=(frame2+frame0)/2
    
    frame0=(frame2).astype(np.int8)


    cv2.putText(frame,txt, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    
    #frame2-=frame2.min()
    #frame2/=frame2.max()
             
    # Display the resulting frame
    cv2.imshow('Webcam',frame)

    cv2.imshow('Filtered Image',frame2)

    
    cv2.imshow('FFT Filtered image',F)

    key=cv2.waitKey(1)
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        cv2.imwrite("screen_{}.png".format(idscreen),frame2*255)
        idscreen+=1
    if (key & 0xFF) in [ ord(' ')]:
        do_fft=not do_fft
    if (key & 0xFF) in [ ord('+')]:
        s+=1     
        cut*=1.2   
    if (key & 0xFF) in [ ord('-')]:
        s=max(1,s-1)   
        cut/=1.2
    if (key & 0xFF) in [ ord('w')]:
        win=not win      
    if (key & 0xFF) in [ ord('f')]:
        idfilt+=1 
        s=1
        cut=0.5
    if (key & 0xFF) in [ ord('i')]:
        idfilt=11
        s=1
        cut=0.5        
    if (key & 0xFF) in [ ord('F')]:
        idfilt-=1         
    if (key & 0xFF) in [ ord('h')]:
        if not half:
            do_tv=True
            half=True
        else:
            half=False
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
