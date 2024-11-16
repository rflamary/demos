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

from sklearn.decomposition import PCA, FastICA, DictionaryLearning, NMF, KernelPCA
from sklearn.cluster import KMeans

models = [
    PCA,
    FastICA,
    KMeans,
    NMF,
    DictionaryLearning,
    KernelPCA,
 ]

models_params=[
        {},
        {},
        {"random_state":42},
        {"random_state":42, "alpha_W":0.1, "alpha_H":0.0},
        {"alpha":1, "max_iter":100,"random_state":42},
        {"kernel":"rbf", "gamma":0.5, "fit_inverse_transform":True},
        ]

# list of models with max number of components to two
limit_2 = [0,1 ]


n_components=2

model_names=['PCA', 'ICA', 'KMeans', 'NMF', 'Sparse DL', 'Kernel PCA' ]

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
pygame.display.set_caption('2D Dictionary Learning demonstration')



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

color_data=(10, 10, 10)

color_traj=(150, 150, 150)

color_blue = (0,0,255)
color_red = (255,0,0)
color_green = (0,255,0)
color_orange = (255,165,0)
color_purple = (128,0,128)
color_yellow = (255,255,0)
color_pink = (255,192,203)

colors = [color_blue, color_red, color_green, color_orange, color_purple, color_yellow, color_pink]
colors_np = np.array(colors, dtype=float)




# prepare texts
font = pygame.font.Font(None, 25) # title
font2 = pygame.font.Font(None, 20) # text
font4 = pygame.font.Font(None, 20)

radius=7
width2=2

width_m=2

scale_G=1.0
scale_dic=100

use_reg=False
reg=1


lst_src=[]
lst_tgt=[]
xp=[]

G=np.zeros((0,0))


def draw_arrow(
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 1,
        head_width: int = 4,
        head_height: int = 2,
    ):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    start = pygame.Vector2(*start)
    end = pygame.Vector2(*end)
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pygame.draw.polygon(surface, color, head_verts)
    pygame.draw.aaline(surface, color, start, end, body_width)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)
        pygame.draw.aaline(surface, color, start, end, body_width)


def draw_source(world,pos, color=color_data):
    #pygame.draw.circle(world,color_src_out, pos, radius+width2)
    #pygame.draw.circle(world,color_src_center, pos, radius)

    # draw a cross
    pygame.draw.line(world,color,(pos[0]-radius,pos[1]),(pos[0]+radius,pos[1]),width_m)
    pygame.draw.line(world,color,(pos[0],pos[1]-radius),(pos[0],pos[1]+radius),width_m)

def draw_center(world,clf, color=color_data):

    pos=clf.mean_[:2]
    pos = pos.astype(int)
    #pygame.draw.circle(world,color_src_out, pos, radius+width2)
    #pygame.draw.circle(world,color_src_center, pos, radius)

    # draw a cross
    pygame.draw.line(world,color,(pos[0]-radius,pos[1]),(pos[0]+radius,pos[1]),width_m+2)
    pygame.draw.line(world,color,(pos[0],pos[1]-radius),(pos[0],pos[1]+radius),width_m+2)

def draw_dictionary(world,clf):
    n = clf.components_.shape[0]
    D = clf.components_
    center = clf.mean_[:2].astype(int)

    
    for i in range(n):
        vec = D[i,:2]
        if model_names[idclass] in ['KMeans']: # NMF of kmenas no scaling
            pass
        else:
            vec = vec/np.linalg.norm(vec)*scale_dic
        vec = vec.astype(int)
        #pygame.draw.circle(world,color_src_out, pos, radius+width2)
        #pygame.draw.circle(world,color_src_center, pos, radius)

        # draw a cross
        draw_arrow(world, center, (center[0]+vec[0],center[1]+vec[1] ), darken_color(colors[i],0.7), body_width=2, head_width=10, head_height=10)

def find_overlap(pos,lst):
    res=-1
    for i,pos2 in enumerate(lst):
        if np.sqrt(np.sum(np.square(np.array(pos)-np.array(pos2))))<radius+width2:
            res=i
    return res
            

def update_method(src):
    xs=np.array(src)
    if model_names[idclass] == 'Sparse DL': # center before fit
        mean = np.mean(xs,axis=0)
        xs = xs-mean
    clf = models[idclass](n_components,**models_params[idclass]).fit(xs)
    xp = clf.transform(xs)
    if model_names[idclass] == 'Sparse DL': # DL does not have inverse_transform nor centering of the data
        clf.mean_= mean
        xpr = xp.dot(clf.components_) + mean[None,:]
    elif model_names[idclass] == 'NMF': # NMF
        clf.mean_ = np.zeros(2)
        xpr = clf.inverse_transform(xp)
    elif model_names[idclass] == 'KMeans': # Kmeans
        clf.mean_ =np.mean(xs,axis=0)
        clf.components_ = clf.cluster_centers_ - clf.mean_[None,:]
        xp = 1.0*(clf.predict(xs)[:,None]==np.arange(clf.n_clusters)[None,:])
        xpr = clf.cluster_centers_[clf.predict(xs)]
    else:  
        xpr = clf.inverse_transform(xp)
    return clf, xp, xpr

def get_text(lst,deltay=0):
    lstf=[]
    lstp=[]
    for i,txt in enumerate(lst):
        lstf.append(font2.render('{}'.format(txt), 1, color_text))
        lstp.append(get_pos(lstf[-1],15+20*i+deltay,5))
    
    return lstf, lstp

def get_color(props, alpha=1):
    d = len(props)
    props =np.abs(props)
    props /= props.sum()
    col = colors_np[:d,:].T.dot(props)
    col = col*alpha +(1-alpha)*255
    return [int(c) for c in col]

def darken_color(color, alpha):
    return [int(c*alpha) for c in color]


lst_txt=['Add point: left click','Remove point: right click','Clear : c','Switch method : m', 'Method : PCA', '', '']

lstf, lstp = get_text(lst_txt)

def update_txt(lstf,reg,use_reg):
    lstf[4]=font2.render('Method : {}'.format(model_names[idclass]), 1, color_text)
    lstf[5]=font2.render('Nb dic. : {}'.format(n_components), 1, color_text)
    # if idclass==1:
    #     lstf[5]=font2.render('Gamma : {:1.2e}'.format(models_params[3]["gamma"]), 1, color_text)
    # else:
    #     lstf[5]=font2.render('', 1, color_text)
    
    
    pass
    

plot_dico=False
plot_proj=False

data[:]=255

update_txt(lstf,reg,use_reg)

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
            if event.key in [pygame.K_s] :
                imgdata = pygame.surfarray.array3d(world)     
                imgdata=imgdata.swapaxes(0,1)
                scipy.misc.imsave('screen_{}.png'.format(datetime.datetime.now()), imgdata)    
            if event.key in [pygame.K_SPACE] :        
                plot_dico= not plot_dico  
            if event.key in [pygame.K_p] :        
                plot_proj= not plot_proj            
            if event.key in [pygame.K_m] :
                idclass= (idclass+1) % len(model_names)
                if idclass in limit_2:
                    n_components=min(2,n_components)
                update_txt(lstf,reg,use_reg)
                if len(lst_src)>=2:
                    clf, xp, xpr =update_method(lst_src)  
            if event.key in [pygame.K_DOWN] :   
                n_components=max(1,n_components-1)
                update_txt(lstf,reg,use_reg)
                if len(lst_src)>=2:
                    clf, xp, xpr = update_method(lst_src)   
            if event.key in [pygame.K_UP] :   
                n_components+=1
                n_components=min(7,n_components)
                if idclass in limit_2:
                    n_components=min(2,n_components)
                update_txt(lstf,reg,use_reg)
                if len(lst_src)>=2:
                    clf, xp, xpr = update_method(lst_src)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            #print(event)
            if event.button==1:
                lst_src.append(pos)
                #print(pos)
            elif event.button==3:
                i=find_overlap(pos,lst_src)
                if not i<0:
                    del lst_src[i]

            if len(lst_src)>=2:
                clf, xp, xpr = update_method(lst_src)


    # if plot_class:
    #     data[:]=(ygrid==1)[:,:,None]*c1+(ygrid==2)[:,:,None]*c2+ (ygrid==0)[:,:,None]*np.ones((1,1,3),dtype=int)*255
    # else:
    #     data[:]=255

    world=pygame.pixelcopy.make_surface(np.swapaxes(data,0,1))

    if plot_proj:
        for i,pos in enumerate(lst_src):
            pos2 = (xpr[i,:2]).astype(int)
            pygame.draw.aaline(world,get_color(xp[i],0.3),pos,pos2,2)
    
    for i,pos in enumerate(lst_src):
        if plot_dico and plot_proj:
            draw_source(world,pos, color=get_color(xp[i],0.5))
            draw_source(world,xpr[i], color=get_color(xp[i],0.7))
        elif plot_dico:
            #print(get_color(xp[i]))
            draw_source(world,pos, color=get_color(xp[i]))
        elif plot_proj:
            draw_source(world,xpr[i], color=get_color(xp[i],0.7))
        else:
            draw_source(world,pos)
    # for pos in lst_tgt:
    #     draw_target(world,pos) 

    # if len(lst_src)>2 and len(lst_tgt)>2:
    #     update_cov(lst_src,lst_tgt)

    if plot_dico and len(lst_src)>2:
        if not  model_names[idclass] in ['Kernel PCA']: 
            draw_center(world,clf)
            draw_dictionary(world,clf)


    screen.blit(world, (0,0))        

    for t,p in zip(lstf, lstp):
        screen.blit(t, p)


    pygame.display.flip() #RENDER WINDOW


