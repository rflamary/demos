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


cap = cv2.VideoCapture(0)

reg=2e-1
alpha=0.2
pred=0
idscreen=0
loop=True
do_tv=False
half=False
noise=False
signoise=0.2
col=[255,1,1]
max_iters=1

tau_noise=0.002

ret, frame0 = cap.read()
frame0=np.asfortranarray(np.array(frame0)/255)


def solve_tv(new,old,reg):
    #return ptv.tv1_2d(new,reg,max_iters=max_iters,n_threads=8)
    temp=np.asfortranarray(old)
    # une itération
    ptv._dr_tv2d(np.asfortranarray(new),reg,temp,max_iters,np.zeros(3),n_threads=4)
    return temp


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    
    frame2=np.array(frame)/255
    frame2=frame2[:,::-1,:]
    #frame2=frame2[::2,::2,:]
    
    if noise:

        # gaussian noise
        #frame2+=signoise*np.random.randn(frame2.shape[0],frame2.shape[1],frame2.shape[2])
        
        # salt and pepper noise
        nz=np.sign(np.random.rand(frame2.shape[0],frame2.shape[1])-.5)*(np.random.rand(frame2.shape[0],frame2.shape[1])<tau_noise)
        
        frame2+=nz[:,:,None]
        frame2=np.clip(frame2,0,1)
    
    if do_tv:
        
        if not half:
            for i in range(3):
                frame2[:,:,i]=solve_tv(frame2[:,:,i],frame0[:,:,i],reg)
        else:
            n2=frame.shape[1]//2
            for i in range(3):
                frame2[:,n2:,i]=solve_tv(frame2[:,n2:,i],frame0[:,n2:,i],reg)
    
    
    #frame2=(frame2+frame0)/2
    
    frame0=frame2
    
    #frame2-=frame2.min()
    #frame2/=frame2.max()
             
    # Display the resulting frame
    cv2.imshow('TV Webcam Demo',frame2)
    key=cv2.waitKey(1)
    if (key & 0xFF) in [ ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        cv2.imwrite("screen_{}.png".format(idscreen),frame2*255)
        idscreen+=1
    if (key & 0xFF) in [ ord(' ')]:
        do_tv=not do_tv
    if (key & 0xFF) in [ ord('+')]:
        reg*=1.5        
    if (key & 0xFF) in [ ord('-')]:
        reg/=1.5           
    if (key & 0xFF) in [ ord('n')]:
        noise=not noise        
    if (key & 0xFF) in [ ord('b')]:
        signoise/=2
        tau_noise/=2
    if (key & 0xFF) in [ ord(',')]:
        signoise*=2
        tau_noise*=2        
    if (key & 0xFF) in [ ord('h')]:
        if not half:
            do_tv=True
            half=True
        else:
            half=False
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
