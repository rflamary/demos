PYTHON Demos
============

This repository contains several python3 scripts demonstrations. They all have
MIT license.

# How to run

All scripts from the demos are available in the `demos` folder. 

One can run easily all demos using the `demo.py` script in the root folder. 

The list of available demos is given by:
```bash
python3 demo.py -l
```

To run the demo you can execute:
```bash
python3 demo.py [demo_name]
```

To print the list of dependencies for a given demo you can run:
```bash
python3 demo.py -d [demo_name]
```
you can store it in a text file and install it with pip.

# Demos

## Total Variation on webcam

In this demonstration we perform online Total Variation regularization on the
video captured on the webcam. 

Screenshot:

![screenshot](data/screen_tv.png "screenshot")

Shortcuts:

* <kbd>q</kbd> : Quit demo
* <kbd>s</kbd> : Save screen to png
* <kbd>s</kbd> : Save screen to png
* <kbd>Space</kbd> : Do full image TV
* <kbd>h</kbd> : Do half image TV (screenshot)
* <kbd>+</kbd> : Make TV regularization stronger
* <kbd>-</kbd> : Make TV regularization weaker
* <kbd>n</kbd> : Add salt and pepper noise
* <kbd>b</kbd> : Make noise more aggressive
* <kbd>,</kbd> : Make noise less aggressive

Dependencies:

This demo uses [opencv-python](https://github.com/skvark/opencv-python) for
webcam access and visualization and [prox_tv](https://github.com/albarji/proxTV) for total variation proximal
operator. 

## Real time audio spectrum

In this demonstration we plot in real time the audio spectrum (and the time
frequency analysis) of the microphone.

![screenshot](data/screen_spectrum.png "screenshot")

Shortcuts:

* <kbd>q</kbd> or <kbd>Esc</kbd>  : Quit demo
* <kbd>Space</kbd>  : Pause/unpause demo
* <kbd>r</kbd>  : Reset time
* <kbd>+</kbd> or <kbd>N</kbd> : Multiply nfft by 2 (zoom in to lower frequencies)
* <kbd>-</kbd> or <kbd>n</kbd>  : Divide nfft by 2 (zoom out)
* <kbd>W</kbd> : Multiply time window size by 2
* <kbd>w</kbd> : Divide time window size by 2
* <kbd>P</kbd>/<kbd>p</kbd> : Change scale of power spectrum (up/down)
* <kbd>S</kbd>/<kbd>s</kbd> : Change scale of power spectrum (up/down)

Dependencies:

This demo uses [pygame](https://www.pygame.org/) for visualization and
[PyAudio](http://people.csail.mit.edu/hubert/pyaudio/) for microphone recording. 

## 2D classification demo

In this demonstration we illustrate the decision function and update of 2D
classifiers when adding training samples from negative or positive classes. We
provide decision functions for Linear and Gaussian Kernel SVMs.

![screenshot](data/screen_classif_2D.png "screenshot")

* <kbd>q</kbd> : Quit demo
* <kbd>left click</kbd> : Add samples from red class
* <kbd>right click</kbd> : Add samples from blue class
* <kbd>c</kbd> : Clear training data
* <kbd>Space</kbd> : Show/hide decision function
* <kbd>m</kbd> : Change classifier (Linear/Gaussian SVM)
* <kbd>Up</kbd> : Make Gaussian Gamma parameter larger
* <kbd>Down</kbd> : Make Gaussian Gamma parameter smaller
* <kbd>s</kbd> : save screenshot

Dependencies:

This demo uses [pygame](https://www.pygame.org/) for visualization and
[Scikit-learn](https://scikit-learn.org/) for classification.


# Installing dependencies

There is an important number of dependencies to run all the demos. Each script
has its own dependencies. 
To print the list of dependencies for a given demo you can run:
```bash
python3 demo.py -d [demo_name]
```

 


