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
import sys, pygame
import scipy.misc
import datetime
import scipy as sp

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    SVC,
    SVC
 ]

classifier_params=[
        {},
        {},
        {"kernel":"linear", "C":0.025},
        {"gamma":5e-5, "C":10},    
        ]

classifier_names=['LDA', 'QDA', 'Linear SVM', 'Gaussian SVM']

idclass=0



# function to detect if key is caps or not (pygame doe not provide it...)
def is_caps():
    return pygame.key.get_mods() & pygame.KMOD_SHIFT or  pygame.key.get_mods() & pygame.KMOD_CAPS

def get_pos(txt,y,x=0):
    pos=txt.get_rect()
    pos.left=x
    pos.centery=y
    return pos

# init pygame
pygame.init()
# screen size
size = width, height = 1024, 768
screen = pygame.display.set_mode(size)
pygame.display.set_caption('2D classifier demonstration')



color_c1=pygame.Color(255,200,200)
color_c2=pygame.Color(200,200,255)

c2=np.array([255,200,200]).reshape((1,1,-1))
c1=np.array([200,200,255]).reshape((1,1,-1))

x1=np.arange(width)
x2=np.arange(height)

X1,X2=np.meshgrid(x1,x2)

xgrid=np.concatenate((X1.ravel()[:,None],X2.ravel()[:,None]),1)
ygrid=np.zeros_like(X1)

# color palette
pal = [(max((x-128)*2,0),x,min(x*2,255)) for x in range(256)]

# background image
world=pygame.Surface((width,height),depth=8) # MAIN SURFACE
#world.set_palette(pal)

data=np.array(np.zeros((height,width,3)),dtype=int)


color_title=(50,50,100)
color_text=(100,100,120)

color_src_center=(250, 0, 0)
color_src_out=(150, 0, 0)

color_tgt_center=(0, 0, 250)
color_tgt_out=(0, 0, 150)

color_G=(255, 255, 255)
# prepare texts
font = pygame.font.Font(None, 25) # title
font2 = pygame.font.Font(None, 20) # text
font4 = pygame.font.Font(None, 20)

radius=7
width2=2

width_m=2

scale_G=1.0

use_reg=False
reg=1


lst_src=[]
lst_tgt=[]

G=np.zeros((0,0))

def draw_cov(C,mu,n,color):

    M=sp.linalg.sqrtm(C)

    angles=2*np.pi*np.linspace(0,1,n)
    v0=np.zeros(2)
    v0[0]=1

    for i in range(n-1):
        pos1=M.dot(np.array([np.cos(angles[i]),np.sin(angles[i])]))+mu
        pos2=M.dot(np.array([np.cos(angles[i+1]),np.sin(angles[i+1])]))+mu
        pygame.draw.line(world,color,pos1,pos2,2)


def update_cov(src,tgt):
    xs=np.array(src)
    xt=np.array(tgt)

    mus=np.mean(xs,0)
    mut=np.mean(xt,0)

    Cs=np.cov(xs.T)
    Ct=np.cov(xt.T)
    C=(Cs*xs.shape[0]+Ct*xt.shape[0])/(xs.shape[0]+xt.shape[0])

    if idclass==0:
        draw_cov(C,mut,100,color_tgt_center)
        draw_cov(C,mus,100,color_src_center)
    elif idclass==1:
        draw_cov(Ct,mut,100,color_tgt_center)
        draw_cov(Cs,mus,100,color_src_center)
    

def draw_source(world,pos):
    pygame.draw.circle(world,color_src_out, pos, radius+width2)
    pygame.draw.circle(world,color_src_center, pos, radius)
    
def draw_target(world,pos):
    pygame.draw.circle(world,color_tgt_out, pos, radius+width2)
    pygame.draw.circle(world,color_tgt_center, pos, radius)    

def find_overlap(pos,lst):
    res=-1
    for i,pos2 in enumerate(lst):
        if np.sqrt(np.sum(np.square(np.array(pos)-np.array(pos2))))<radius+width2:
            res=i
    return res
            

def update_classif(src,tgt):
    xs=np.array(src)
    xt=np.array(tgt)
    
    xtot=np.concatenate((xs,xt),0)
    ytot=np.concatenate((np.ones(xs.shape[0]),2*np.ones(xt.shape[0])),0)
    
    
    
    return classifiers[idclass](**classifier_params[idclass]).fit(xtot,ytot)

def get_text(lst,deltay=0):
    lstf=[]
    lstp=[]
    for i,txt in enumerate(lst):
        lstf.append(font2.render('{}'.format(txt), 1, color_text))
        lstp.append(get_pos(lstf[-1],15+20*i+deltay,5))
    
    return lstf, lstp

lst_txt=['Class 1: left click','Class 2 : right click','Clear : c','Switch method : m', 'Classifier: LDA', '']

lstf, lstp = get_text(lst_txt)

def update_txt(lstf,reg,use_reg):
    lstf[4]=font2.render('Classifier : {}'.format(classifier_names[idclass]), 1, color_text)
    if idclass==1:
        lstf[5]=font2.render('Gamma : {:1.2e}'.format(classifier_params[3]["gamma"]), 1, color_text)
    else:
        lstf[5]=font2.render('', 1, color_text)
    
    
    pass
    

plot_class=False

while 1:

    # keyboard events
    for event in pygame.event.get(): #check if we need to exit
        if event.type == pygame.QUIT:
            pygame.quit();
            sys.exit()
        if event.type == pygame.KEYDOWN:
            #print event.key
            if event.key in [pygame.K_ESCAPE,pygame.K_q] :
                pygame.quit()
                sys.exit()
            if event.key in [pygame.K_c] :
                lst_tgt=[]
                lst_src=[]
                ygrid=np.zeros_like(X1)
            if event.key in [pygame.K_s] :
                imgdata = pygame.surfarray.array3d(world)     
                imgdata=imgdata.swapaxes(0,1)
                scipy.misc.imsave('screen_{}.png'.format(datetime.datetime.now()), imgdata)    
            if event.key in [pygame.K_SPACE] :        
                plot_class= not plot_class  
            if event.key in [pygame.K_m] :
                idclass= (idclass+1) % len(classifier_names)
                update_txt(lstf,reg,use_reg)
                clf=update_classif(lst_tgt,lst_src)
                ygrid=clf.predict(xgrid).reshape((height,width))     
            if event.key in [pygame.K_DOWN] :   
                classifier_params[3]["gamma"]/=1.5
                update_txt(lstf,reg,use_reg)
                clf=update_classif(lst_tgt,lst_src)
                ygrid=clf.predict(xgrid).reshape((height,width))     
            if event.key in [pygame.K_UP] :   
                classifier_params[32]["gamma"]*=1.5
                update_txt(lstf,reg,use_reg)
                clf=update_classif(lst_tgt,lst_src)
                ygrid=clf.predict(xgrid).reshape((height,width))   
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            #print(event)
            if event.button==1:
                i=find_overlap(pos,lst_src)
                if i<0:
                    lst_src.append(pos)
                else:
                    del lst_src[i]
            elif event.button==3:
                i=find_overlap(pos,lst_tgt)
                if i<0:
                    lst_tgt.append(pos)
                else:
                    del lst_tgt[i]
                    
            if lst_tgt and lst_src:
                clf=update_classif(lst_tgt,lst_src)
                ygrid=clf.predict(xgrid).reshape((height,width))
                

    if plot_class:
        data[:]=(ygrid==1)[:,:,None]*c1+(ygrid==2)[:,:,None]*c2+ (ygrid==0)[:,:,None]*np.ones((1,1,3),dtype=int)*255
    else:
        data[:]=255

    world=pygame.pixelcopy.make_surface(np.swapaxes(data,0,1))
    
    for pos in lst_src:
        draw_source(world,pos)
    for pos in lst_tgt:
        draw_target(world,pos) 

    if len(lst_src)>2 and len(lst_tgt)>2:
        update_cov(lst_src,lst_tgt)


    screen.blit(world, (0,0))        

    for t,p in zip(lstf, lstp):
        screen.blit(t, p)


    pygame.display.flip() #RENDER WINDOW


