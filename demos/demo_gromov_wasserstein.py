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

#import pyaudio
import numpy as np
import sys, pygame
import ot
import random
import scipy.misc
import datetime
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
pygame.display.set_caption('Discrete Gromov-Wasserstein demonstration')




# color palette
pal = [(max((x-128)*2,0),x,min(x*2,255)) for x in range(256)]

# background image
world=pygame.Surface((width,height), pygame.SRCALPHA) # MAIN SURFACE
#world.set_palette(pal)

data=np.array(np.zeros((height,width,4)),dtype=int)


color_title=(50,50,100)
color_text=(100,100,120)

color_src_center=(250, 0, 0)
color_src_out=(150, 0, 0)
color_src_pairwise=(250, 50, 50)


color_tgt_center=(0, 0, 250)
color_tgt_out=(0, 0, 150)
color_tgt_pairwise=(50, 50, 250)


color_G=(0, 0, 0)

show_graph = False
d_graph= 100


# prepare texts
font = pygame.font.Font(None, 25) # title
font2 = pygame.font.Font(None, 20) # text
font4 = pygame.font.Font(None, 20)

radius=7
width=2

width_m=2

scale_G=0.7

use_reg=False
reg=1


lst_src=[]
lst_tgt=[]

G=np.zeros((0,0))

def get_color(s,color=None):
    if color is None:
        color=color_G
    col=pygame.Color(*color)
    H, S, L, A = col.hsla

    temp = pygame.Color(255,255,255)
    temp.hsla = (H, S, ((1-s)*95+s*L), A)

    #return pygame.Color(*[int(min(255*(1-s)+s*c,255)) for c in color])
    #return pygame.Color(255,255,255).lerp(color,s)
    return temp

def draw_source(world,pos):
    pygame.draw.circle(world,color_src_out, pos, radius+width)
    pygame.draw.circle(world,color_src_center, pos, radius)
    
def draw_target(world,pos):
    pygame.draw.circle(world,color_tgt_out, pos, radius+width)
    pygame.draw.circle(world,color_tgt_center, pos, radius)    

def find_overlap(pos,lst):
    res=-1
    for i,pos2 in enumerate(lst):
        if np.sqrt(np.sum(np.square(np.array(pos)-np.array(pos2))))<radius+width:
            res=i
    return res
            

def update_G(src,tgt):
    xs=np.array(src)
    xt=np.array(tgt)
    C1=ot.dist(xs,xs,'euclidean')
    C2=ot.dist(xt,xt,'euclidean')
    if False:
        C1=1.0*(C1<d_graph)
        G2=1.0*(C2<d_graph)
    if not use_reg:
        return ot.gromov.gromov_wasserstein(C1,C2,ot.unif(C1.shape[0]),ot.unif(C2.shape[0]),'square_loss' )
    else:
        return ot.gromov.entropic_gromov_wasserstein(C1,C2,ot.unif(C1.shape[0]),ot.unif(C2.shape[0]),'square_loss',reg)



def draw_G(world,G,src,tgt):
    Gmax=G.max()
    isort = np.argsort(G.ravel())
    for k in isort:
        i,j = np.unravel_index(k,G.shape)
        if G[i,j]>1e-8:
            scale= G[i,j]/Gmax
            #print([int(255*(scale)) for c in color_G])
            pygame.draw.line(world,get_color(scale),src[i],tgt[j],width_m) #

def draw_pairwise2(world,src,tgt):
    for x1 in src:
        for x2 in src:
            pygame.draw.line(world,color_src_pairwise,x1,x2,width_m) #
                #pygame.draw.line(world,color_G,src[i],tgt[j],int(width_m*+1)) #
    for x1 in tgt:
        for x2 in tgt:
            pygame.draw.line(world,color_tgt_pairwise,x1,x2,width_m) #



def draw_pairwise(world,src,tgt):
    if len(src)>2 and not show_graph:
        xs=np.array(src)
        C1=ot.dist(xs,xs,'euclidean')
        C1min,C1max = C1.min(),C1.max()
        isort = np.argsort(-C1.ravel())
        for k in isort:
            i,j = np.unravel_index(k,C1.shape)
            x1,x2= src[i],src[j]
            s=((C1max-C1[i,j])/(C1max-C1min))**1.5
               # draw_line_alpha(world,get_color(s,color_src_pairwise)  ,x1,x2,width_m)
            pygame.draw.line(world,get_color(s,color_src_pairwise)  ,x1,x2,width_m) #
    else:
        if len(src)>1:
            xs=np.array(src)
            C1=ot.dist(xs,xs,'euclidean')
            for i,x1 in enumerate(src):
                for j,x2 in enumerate(src):
                    if not show_graph or C1[i,j]<d_graph:
                        pygame.draw.line(world,color_src_pairwise,x1,x2,width_m) #
    
    if len(tgt)>2 and not show_graph:
        xs=np.array(tgt)
        C1=ot.dist(xs,xs,'euclidean')
        C1min,C1max = C1.min(),C1.max()
        isort = np.argsort(-C1.ravel())
        for k in isort:
            i,j = np.unravel_index(k,C1.shape)
            x1,x2= tgt[i],tgt[j]
            s=((C1max-C1[i,j])/(C1max-C1min))**1.5
               # draw_line_alpha(world,get_color(s,color_src_pairwise)  ,x1,x2,width_m)
            pygame.draw.line(world,get_color(s,color_tgt_pairwise)  ,x1,x2,width_m) #
    else:
        if len(tgt)>1:
            xs=np.array(tgt)
            C1=ot.dist(xs,xs,'euclidean')
            for i,x1 in enumerate(tgt):
                for j,x2 in enumerate(tgt):
                    if not show_graph or C1[i,j]<d_graph:
                        pygame.draw.line(world,color_tgt_pairwise,x1,x2,width_m) #


def rotate_list(lst,theta):
    x=np.array(lst)
    mx=x.mean(0,keepdims=True)
    R=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    xr=(x-mx).dot(R)+mx
    xr=xr.astype(int)
    return xr.tolist()

def get_text(lst,deltay=0):
    lstf=[]
    lstp=[]
    for i,txt in enumerate(lst):
        lstf.append(font2.render('{}'.format(txt), 1, color_text))
        lstp.append(get_pos(lstf[-1],15+20*i+deltay,5))
    
    return lstf, lstp

lst_txt=['Source: left click','Target : right click','Clear : c','Switch Reg OT : r', 'Reg value : {}'.format(0)]

lstf, lstp = get_text(lst_txt)

def update_txt(lstf,reg,use_reg):
    
    if use_reg:
        lstf[4]=font2.render('Reg value : {:1.2e}'.format(reg), 1, color_text)
    else:
        lstf[4]=font2.render('Reg value : 0'.format(reg), 1, color_text)
        
plot_G=False  


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
            if event.key in [pygame.K_g] :
                show_graph = not show_graph
            if event.key in [pygame.K_t] :
                if lst_tgt:
                    lst_tgt=rotate_list(lst_tgt,.1)
            if lst_tgt and lst_src:
                    G=update_G(lst_src,lst_tgt)        
            if event.key in [pygame.K_s] :
                imgdata = pygame.surfarray.array3d(world)     
                imgdata=imgdata.swapaxes(0,1)
                scipy.misc.imsave('screen_{}.png'.format(datetime.datetime.now()), imgdata)
            if event.key in [pygame.K_r] :
                use_reg=not use_reg  
                update_txt(lstf,reg,use_reg)
                if lst_tgt and lst_src:
                    G=update_G(lst_src,lst_tgt)      
            if event.key in [pygame.K_2, pygame.K_KP2,pygame.K_p] :
                reg*=1.5      
                print(reg)
                update_txt(lstf,reg,use_reg)
                if lst_tgt and lst_src:
                    G=update_G(lst_src,lst_tgt)                      
            if event.key in [pygame.K_1, pygame.K_KP1,pygame.K_m] :
                reg/=1.5      
                print(reg)
                update_txt(lstf,reg,use_reg)
                if lst_tgt and lst_src:
                    G=update_G(lst_src,lst_tgt)                      
            if event.key in [pygame.K_e] :
                imax=min(len(lst_tgt),len(lst_src))
                random.shuffle(lst_tgt)
                random.shuffle(lst_src)
                lst_tgt=lst_tgt[:imax]
                lst_src=lst_src[:imax]
                if lst_tgt and lst_src:
                    G=update_G(lst_src,lst_tgt)                
            if event.key in [pygame.K_SPACE] :        
                plot_G= not plot_G                   
                
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
                G=update_G(lst_src,lst_tgt)

 
    data[:]=255
    
    # appedn data to the window
    #pygame.surfarray.blit_array(world,data.T) #place data in window
    world.fill((255,255,255))
    
    
    if lst_tgt or lst_src:
        draw_pairwise(world,lst_src,lst_tgt)

    if plot_G:
        # print all texts

        if lst_tgt and lst_src:
            draw_G(world,G,lst_src,lst_tgt)
    
    for pos in lst_src:
        draw_source(world,pos)
    for pos in lst_tgt:
        draw_target(world,pos) 

    screen.blit(world, (0,0))        
#    screen.blit(txtpower, txtpowerpos)
#    screen.blit(txtspg, txtspgpos)
#    screen.blit(tnfft, pnfft)
    for t,p in zip(lstf, lstp):
        screen.blit(t, p)
#    screen.blit(tpscale2, ppscale2)
#    screen.blit(tpscale3, ppscale3)
#    screen.blit(tpreg, ppreg)
#    screen.blit(tfmax, pfmax)
#    screen.blit(tspec, pspec)
#    screen.blit(tunmix, ptunmix)

    pygame.display.flip() #RENDER WINDOW


