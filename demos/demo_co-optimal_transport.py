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
import ot
import random
import scipy.misc
import datetime
np.set_printoptions(suppress=False,precision=1, formatter= {'all':lambda x: '{:1.1f}'.format(x)})

def init_matrix_np(C1, C2, p, q):
    def f1(a):
        return (a ** 2)

    def f2(b):
        return (b ** 2)

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(f1(C2).shape[0]).reshape(1, -1))
    constC2 = np.dot(np.ones(f1(C1).shape[0]).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))

    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2

def cot_numpy(P1, P2, w1 = None, w2 = None, v1 = None, v2 = None, niter=10, algo='emd', reg=0,algo2='emd', reg2=0,
              eps=1e-16, C_lin=None,
              verbose=True, log=False, random_init=False,force_same=False,batch_size=3,lr=0.1):
    """
    P1 is (n,d1)
    P2 is (m,d2)
    Tv is (n,m)
    Tc is (d1,d2)
    """
    if v1 is None:
        pc = np.ones(P1.shape[1]) / P1.shape[1]  # is (d1,)
    else:
        pc =v1
    if v2 is None:
        qc = np.ones(P2.shape[1]) / P2.shape[1]  # is (d2,)
    else:
        qc = v2
    if w1 is None:
        pv = np.ones(P1.shape[0]) / P1.shape[0]  # is (m,)
    else:
        pv = w1
    if w2 is None:
        qv = np.ones(P2.shape[0]) / P2.shape[0]  # is (n,)
    else:
        qv = w2

    
    Tv = np.ones((P1.shape[0], P2.shape[0])) / (P1.shape[0] * P2.shape[0])  # is (n,m)
    Tc = np.ones((P1.shape[1], P2.shape[1])) / (P1.shape[1] * P2.shape[1])  # is (d1,d2)


    constC_v, hC1_v, hC2_v = init_matrix_np(P1, P2, pc, qc)

    constC_c, hC1_c, hC2_c = init_matrix_np(P1.T, P2.T, pv, qv)
    cost = np.inf

    log_out ={}
    log_out['cost'] = []
    
    for i in range(niter):
        Tvold = Tv
        Tcold = Tc
        costold = cost

        M = constC_v - np.dot(hC1_v, Tc).dot(hC2_v.T)
            
        if C_lin is not None:
            M=M+C_lin

        if algo == 'emd':
            Tv = ot.emd(pv, qv, M, numItermax=1e7)
        elif algo == 'sinkhorn':
            Tv = ot.sinkhorn(pv, qv, M, reg)
        elif algo=='stochastic':
            Tv=ot.stochastic.solve_dual_entropic(pv,qv,M,reg,batch_size=batch_size,lr=lr)
        if force_same:
            Tc=Tv
        else:
            M = constC_c - np.dot(hC1_c, Tv).dot(hC2_c.T)
            if algo2 == 'emd':
                Tc = ot.emd(pc, qc, M, numItermax=1e7)
            elif algo2 == 'sinkhorn':
                Tc = ot.sinkhorn(pc, qc, M, reg2)
            elif algo2=='stochastic':
                Tc = ot.stochastic.solve_dual_entropic(pc, qc, M,reg,batch_size=batch_size,lr=lr)

        delta = np.linalg.norm(Tv - Tvold) + np.linalg.norm(Tc - Tcold)
        cost = np.sum(M * Tc)
        
        if log:
            log_out['cost'].append(cost)
            
        if verbose:
            print('Delta: {0}  Loss: {1}'.format(delta, cost))

        if delta < eps or np.abs(costold - cost) < 1e-7:
            if verbose:
                print('converged at iter ', i)
            break
    print(Tc)
    if log:
        return Tv, Tc, cost, log_out
    else:
        return Tv, Tc, cost


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
pygame.display.set_caption('CO-Optimal Transport demonstration')




# color palette
pal = [(max((x-128)*2,0),x,min(x*2,255)) for x in range(256)]

# background image
world=pygame.Surface((width,height),pygame.SRCALPHA) # MAIN SURFACE
#world.set_palette(pal)

data=np.array(np.zeros((height,width)),dtype=int)


color_title=(50,50,100)
color_text=(100,100,120)

color_src_center=(250, 0, 0)
color_src_out=(150, 0, 0)

color_src_center2=(250, 100, 100)
color_src_out2=(250, 100, 100)

color_tgt_center=(0, 0, 250)
color_tgt_out=(0, 0, 150)

color_tgt_center0=(150, 150, 250)
color_tgt_out0=(150, 150, 200)

color_G=(255, 255, 255)
# prepare texts
font = pygame.font.Font(None, 25) # title
font2 = pygame.font.Font(None, 20) # text
font4 = pygame.font.Font(None, 20)

radius=7
width=2

width_m=2

scale_G=1.0

use_reg=False
proj=False
reg=1


lst_src=[]
lst_tgt=[]

G=np.zeros((0,0))
Gv=np.eye(2)/2
cost=0

def draw_source(world,pos):
    pygame.draw.circle(world,color_src_out, pos, radius+width)
    pygame.draw.circle(world,color_src_center, pos, radius)
    
def draw_target(world,pos):
    pygame.draw.circle(world,color_tgt_out, pos, radius+width)
    pygame.draw.circle(world,color_tgt_center, pos, radius)    

def draw_target0(world,pos):
    pygame.draw.circle(world,color_tgt_out0, pos, radius+width)
    pygame.draw.circle(world,color_tgt_center0, pos, radius)    

def find_overlap(pos,lst):
    res=-1
    for i,pos2 in enumerate(lst):
        if np.sqrt(np.sum(np.square(np.array(pos)-np.array(pos2))))<radius+width:
            res=i
    return res
            
def rotate_list(lst,theta):
    x=np.array(lst)
    mx=x.mean(0,keepdims=True)
    R=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    xr=(x-mx).dot(R)+mx
    xr=xr.astype(int)
    return xr.tolist()

def proj_point(pos,Gv):
    return (int(2*(Gv[0,0]*pos[0]+Gv[0,1]*pos[1])),int(2*(Gv[1,0]*pos[0]+Gv[1,1]*pos[1])))

def update_G(src,tgt):
    xs=np.array(src)
    xt=np.array(tgt)
    
    if not use_reg:
        return cot_numpy(xs,xt)
    else:
        return cot_numpy(xs,xt,algo='sinkhorn',algo2='sinkhorn',reg=reg,reg2=reg)

def draw_G(world,G,src,tgt,Gv):
    Gmax=G.max()
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i,j]>1e-8:
                scale= ((1-G[i,j]/Gmax)*scale_G)+(1-scale_G)
                #print([int(255*(scale)) for c in color_G])
                if proj:
                    pygame.draw.line(world,[int(255*(scale)) for c in color_G],src[i],proj_point(tgt[j],Gv),width_m)
                else:
                    pygame.draw.line(world,[int(255*(scale)) for c in color_G],src[i],tgt[j],width_m) #
                #pygame.draw.line(world,color_G,src[i],tgt[j],int(width_m*+1)) #

def draw_proj(world,tgt,Gv):
    for pos in tgt:
        pygame.draw.line(world,(200,200,200),pos,proj_point(pos,Gv),width_m)

def get_text(lst,deltay=0):
    lstf=[]
    lstp=[]
    for i,txt in enumerate(lst):
        lstf.append(font2.render('{}'.format(txt), 1, color_text))
        lstp.append(get_pos(lstf[-1],15+20*i+deltay,5))
    
    return lstf, lstp

lst_txt=['Source: left click','Target : right click','Clear : c','Switch Reg OT : r', 'Reg value : {}'.format(0), 'Visu : orig', "OT Plan variables :", " ", " " ]#, 'Variable OT matrix: [[{},{}],[{},{}]]'.format(Gv[0,0],Gv[0,1],Gv[1,0],Gv[1,1]) ]
plot_G=False

lstf, lstp = get_text(lst_txt)

def update_txt(lstf,reg,use_reg,proj):
    
    if use_reg:
        lstf[4]=font2.render('Reg value : {:1.2e}'.format(reg), 1, color_text)
    else:
        lstf[4]=font2.render('Reg value : 0'.format(reg), 1, color_text)

    if proj:
        lstf[5]=font2.render('Visu : orig+projected', 1, color_text)
    else:
        lstf[5]=font2.render('Visu : orig', 1, color_text)

    if plot_G:
        txt0=str(Gv).split('\n')
        lstf[7]=font2.render(txt0[0], 1, color_text)
        if len(txt0)>1 : lstf[8]=font2.render(txt0[1], 1, color_text)
    else:
        lstf[6]=font2.render('OT Plan variables :', 1, color_text)

        lstf[7]=font2.render(' ', 1, color_text)
        lstf[8]=font2.render(' ', 1, color_text)
    

        



while 1:

    # keyboard events
    for event in pygame.event.get(): #check if we need to exit
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            #print event.key
            if event.key in [pygame.K_ESCAPE,pygame.K_q] :
                pygame.quit()
                sys.exit()
            if event.key in [pygame.K_c] :
                lst_tgt=[]
                lst_src=[]
                update_txt(lstf,reg,use_reg,proj)  
            if event.key in [pygame.K_t] :
                if lst_tgt:
                    lst_tgt=rotate_list(lst_tgt,.1)
                if lst_tgt and lst_src:
                    G,Gv,cost=update_G(lst_src,lst_tgt)
                update_txt(lstf,reg,use_reg,proj)
            if event.key in [pygame.K_s] :
                imgdata = pygame.surfarray.array3d(world)     
                imgdata=imgdata.swapaxes(0,1)
                scipy.misc.imsave('screen_{}.png'.format(datetime.datetime.now()), imgdata)
            if event.key in [pygame.K_r] :
                use_reg=not use_reg  
                if lst_tgt and lst_src:
                    G,Gv,cost=update_G(lst_src,lst_tgt)     
                update_txt(lstf,reg,use_reg,proj) 
            if event.key in [pygame.K_2, pygame.K_KP2,pygame.K_p] :
                reg*=1.5      
                print(reg)

                if lst_tgt and lst_src:
                    G,Gv,cost=update_G(lst_src,lst_tgt)        
                update_txt(lstf,reg,use_reg,proj)              
            if event.key in [pygame.K_1, pygame.K_KP1,pygame.K_m] :
                reg/=1.5      
                print(reg)
                update_txt(lstf,reg,use_reg,proj)
                if lst_tgt and lst_src:
                    G,Gv,cost=update_G(lst_src,lst_tgt)    
            if event.key in [pygame.K_SPACE] :        
                plot_G= not plot_G   
                update_txt(lstf,reg,use_reg,proj)           
            if event.key in [pygame.K_a] :        
                proj= not proj     
                update_txt(lstf,reg,use_reg,proj)
            if event.key in [pygame.K_e] :
                imax=min(len(lst_tgt),len(lst_src))
                random.shuffle(lst_tgt)
                random.shuffle(lst_src)
                lst_tgt=lst_tgt[:imax]
                lst_src=lst_src[:imax]
                if lst_tgt and lst_src:
                    G,Gv,cost=update_G(lst_src,lst_tgt)      
                update_txt(lstf,reg,use_reg,proj)          
                
                
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
                G,Gv[:],cost=update_G(lst_src,lst_tgt)
            update_txt(lstf,reg,use_reg,proj)

 
    data[:]=255

    world.fill((255,255,255))
    
    # appedn data to the window
    #pygame.surfarray.blit_array(world,data.T) #place data in window
    
    # print all texts
    if plot_G:
        if lst_tgt and lst_src:
            draw_G(world,G,lst_src,lst_tgt,Gv)

    if proj:
        draw_proj(world,lst_tgt,Gv)
    
    for pos in lst_src:
        draw_source(world,pos)


    if proj:
        for pos in lst_tgt:
            draw_target0(world,pos)

    for pos in lst_tgt:
        if proj:
            draw_target(world,proj_point(pos,Gv))
        else:
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


