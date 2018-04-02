# Video Transition Detector v0.0.1

## Introduction

This is a tool built on python3.6 for video transition detection.

Currently we have made atuo detection for wipe transition.

We apply pipenv to manage the python package below:

- opencv: for image and video processing
- matplotlib: for linear regression debug
- numpy: for matrix manipulation
- PIL: for showing image from opencv in tkinter
- tkinter: for GUI
   
## Build Guide
```bash
# install depend packages
pipenv install
# run program
python3 main.py
```

## Presentation Video - Bob

## Feature list
- [ ] Video Transition Detection Based on **Spatio Temporal Image (STI)**
- [ ] **Copy pixel** 
    + Generate STI with middle column / row from origin frame of videl.
- [ ] **Histogram Difference**
    + Apply chromatic color sampling
    + Apply histogram mechanism
        * Histogram Intersection Method
        * IBM's Histogram Distance Method
    + Apply linear regression for auto detection
  

## Screenshots of Final Product - Bob

## Implementation Details

### STI based on Copy Pixel

The implementation of copy-pixel is to build up STI, whose number of cols is the number of frames and rows is the copy of mid row/col of frame,
when wipe transition happens in video, say horizontal wipe, we can see an apparent gap in STI if we sample the mid col.

<img src=".\screenshot\copypixel\v2_leftwipe_colSampleing.png" width="60%">

It is easy but strongly relies on human's visual recognition.

### STI based on Histogram Difference

#### TO CHROMATIC - LEO

#### TO HISTOGRAM - LEO

#### Intersection Method - LEO

#### IBM's Method - Bob
This method is so-called **HQDM**(Histogram Quadratic Distance Measures). The main idea of it is to calculate the distance between two histograms, hq and ht by using the equation
$$ D_{HQDM}(q,t)=(h_q - h_t)^T A (h_q - h_t)$$
Here A is a predefined matrix where aij denotes the color  the similarity between color of bin i and color of bin j. It is a symmetrical metrix with all 1 on diagonal and becomes smaller to end of rows and cols. Obviously, The more similarity, the more weight of count difference contributes to the histograms difference.

In this way, for every frame(larger than 0), we. 

#### Linear Regression - Bob

## Problems and solutions
### opencv
OpenCV store pixels as BGR. Careful when sampling the pixel color.
### pyplot
RuntimeError for using matpotlib on Mac
```bash
RuntimeError: Python is not installed as a framework. 
The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. 
See the Python documentation for more information on installing Python as a framework on Mac OS X. 
Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. 
See 'Working with Matplotlib on OSX' inthe Matplotlib FAQ for more information.
```
Solution: [relevant github_issue](https://github.com/MTG/sms-tools/issues/36):
```python
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
```
And there iscConflict between matplotlib && cv2.imshow. The RuntimeError above occur even if solution applied. Try to use only one of them (matplotlib) to show some image.




## Discussion

### About Project

### Further Improvement