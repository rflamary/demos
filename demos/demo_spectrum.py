#!/usr/bin/python3
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

import pyaudio
import numpy as np
import sys, pygame
import scipy.fftpack


def get_pos(txt,y,x=0):
    pos=txt.get_rect()
    pos.left=x
    pos.centery=y
    return pos

# print methods texts (with a x for selected method)
def get_txtmethods(method,methodsname):
    tmethods=[]
    pmethods=[]
    for i,name in enumerate(methodsname):
        if i==method:
            txt='[{}] {}'.format('x',name)
        else:
            txt='[{}] {}'.format('  ',name)
        temp=font4.render(txt, 1, color_text)
        tmethods.append(temp)
        ptemp=get_pos(temp,i*20+40)
        pmethods.append(ptemp)
    return tmethods,pmethods

# function to detect if key is caps or not (pygame doe not provide it...)
def is_caps():
    return pygame.key.get_mods() & pygame.KMOD_SHIFT or  pygame.key.get_mods() & pygame.KMOD_CAPS


CHUNK = 4096
WIDTH = 2
CHANNELS = 1
RATE = 44100


# init pygame
pygame.init()
# screen size
size = width, height = 1024, 768
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Periodogram demonstration')




# color palette
pal = [(max((x-128)*2,0),x,min(x*2,255)) for x in range(256)]

# background image
world=pygame.Surface((width,height),depth=8) # MAIN SURFACE
world.set_palette(pal)

data=np.array(np.zeros((height,width)),dtype=int)


# column start (width of left texts)
cstart=150

# current col
col=cstart

# various scalings
sc_pow=20
sc_spec=100
sc_prop0=2.5

maxpower=167
nfft=CHUNK
nfftvisu=400
nspec=200


pmax=0
pw=0;

fmax=nfftvisu*RATE/nfft


color_title=(150,150,255)
color_text=(200,200,255)

# prepare texts
font = pygame.font.Font(None, 25) # title
font2 = pygame.font.Font(None, 20) # text
font4 = pygame.font.Font(None, 20)


# all screen texts
txtpower=font.render('Power', 1, color_title)
txtpowerpos = get_pos(txtpower,height-maxpower+20)

# power scale
tpscale=font2.render('Scale={} (P/p)'.format(sc_pow), 1, color_text)
ppscale = get_pos(tpscale,height-maxpower+40)

#spetrogram
txtspg=font.render('Spectrogram', 1, color_title)
txtspgpos = get_pos(txtspg,height-nfftvisu-maxpower+20)

tfmax=font2.render('Fmax={:5.1f} Hz'.format(fmax), 1, color_text)
pfmax = get_pos(tfmax,height-nfftvisu-maxpower+40)


tnfft=font2.render('NFFT={} (N/n)'.format(nfft), 1, color_text)
pnfft = get_pos(tnfft,height-nfftvisu-maxpower+60)

tpscale2=font2.render('Scale={} (S/s)'.format(sc_spec), 1, color_text)
ppscale2 = get_pos(tpscale2,height-nfftvisu-maxpower+80)


# Spectrum
tspec=font.render('Spectrum', 1, color_title)
pspec = get_pos(tspec,height-nfftvisu-maxpower-nspec+20)




rho=0.2 # scaling spectrum
old_spec=[]



step=3 # step for printing unmixing
pause=0 # pause (or not)


# init pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=CHUNK)


while 1:

    # keyboard events
    for event in pygame.event.get(): #check if we need to exit
        if event.type == pygame.QUIT:
            pygame.quit();
            stream.stop_stream()
            stream.close()
            p.terminate()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            #print event.key
            if event.key in [pygame.K_ESCAPE,pygame.K_q] :
                pygame.quit
                stream.stop_stream()
                stream.close()
                p.terminate()
                sys.exit()
            if event.key in [pygame.K_PLUS,270] :
                nfft*=2
                print('nfft:', nfft)
            if event.key in [pygame.K_MINUS,269] :
                if nfft>nfftvisu*2:
                    nfft/=2
                print( 'nfft:',nfft)
            if event.key in [pygame.K_n] :
                if is_caps():
                    nfft*=2
                elif nfft>nfftvisu*2:
                    nfft/=2
                CHUNK=nfft
                print( 'nfft:',nfft)
            if event.key in [pygame.K_w] :
                if is_caps():
                    CHUNK*=2
                elif CHUNK>512:
                    CHUNK/=2
                nfft=CHUNK
                print('frame:',CHUNK)

            if event.key in [pygame.K_p] :
                if is_caps():
                    sc_pow*=1.5
                else:
                    sc_pow/=1.5
                print('Pow scale:',sc_pow)

            if event.key in [pygame.K_b] :
                pmax=pw
                print('Pow bkgrnd:',pmax)

            if event.key in [pygame.K_s] :
                if is_caps():
                    sc_spec*=1.2
                else:
                    sc_spec/=1.2
            if event.key in [pygame.K_SPACE] :
                if pause:
                    pause=False
                else:
                    pause=True
                    print('Pause')


            if event.key in [pygame.K_r] :
                col=cstart
                data[:,:]=0
                print('reset!')

            # update text
            fmax=nfftvisu*RATE/nfft
            tnfft=font2.render('NFFT={} (N/n)'.format(nfft), 1, color_text)
            tpscale=font2.render('Scale={:4.1f} (P/p)'.format(sc_pow), 1, color_text)
            tfmax=font2.render('Fmax={:5.1f} Hz'.format(fmax), 1, color_text)
            tpscale2=font2.render('Scale={:4.1f} (S/s)'.format(sc_spec), 1, color_text)


    dt = stream.read(CHUNK)
    sig = (np.fromstring(dt, dtype=np.int16))*1.0/32768
    sig-=sig.mean()

    if not pause:

        pw0=np.sum(sig**2)

        #pmax=max(pw0,pmax)
        pw=np.log10(pw0)+3
        if np.isnan(pw):
            pw=0
        if pmax==0:
            pmax=pw
            print('Pow bkgrnd:',pmax)

        data[:,col:col+step]=0
        if (pw*sc_pow)>0:
            data[-max(int(pw*sc_pow),0):,col:col+step]=255
            
            
        # curent cursor
        if col<width-1:
            data[-maxpower:,col+step]=128

        # print spetrogram
        S=scipy.fftpack.fft(sig,nfft)/nfft
        spec=(np.log10(np.abs(S[:nfftvisu])+1e-10)+4)*sc_spec
        specv=(np.log10(np.abs(S)+1e-10)+4.5)*sc_spec

        for t in range(step):
            data[-maxpower-nfftvisu:-maxpower,col+t]=np.minimum(np.maximum(spec[::-1].astype(int),0),255)
        if col<width-1:
            data[-maxpower-nfftvisu:-maxpower,col+step]=128

        # print spectrum
        if len(specv)==len(old_spec):
            spec2=rho*specv+(1.0-rho)*old_spec
        else:
            spec2=  specv

        old_spec=spec2.copy()
        for i in range(min(nfft/2,width-cstart)):
            data[-maxpower-nfftvisu-nspec:-maxpower-nfftvisu-1,cstart+i]=0
            if int(spec2[i]*0.7*(nspec)/255)>0:
                data[-int(spec2[i]*0.7*(nspec)/255)-maxpower-nfftvisu:-maxpower-nfftvisu,cstart+i]=255

    

        # increment current col by step
        col+=step
        if col>=width-step:
            col=cstart


    # add separators
    data[:,cstart]=128
    data[-maxpower,:]=128 # power horz line
    data[-maxpower-nfftvisu-1,:]=128 # spectrum horz line
    data[-maxpower-nfftvisu-nspec-1,:]=128 # spectrum horz line

    # appedn data to the window
    pygame.surfarray.blit_array(world,data.T) #place data in window


    # print all texts
    screen.blit(world, (0,0))
    screen.blit(txtpower, txtpowerpos)
    screen.blit(txtspg, txtspgpos)
    screen.blit(tnfft, pnfft)
    screen.blit(tpscale, ppscale)
    screen.blit(tpscale2, ppscale2)
    screen.blit(tfmax, pfmax)
    screen.blit(tspec, pspec)
    pygame.display.flip() #RENDER WINDOW


