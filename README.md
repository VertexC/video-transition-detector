# video-transition-detector
detect wipe transition in video based on python opencv

# build & run
```shell
# install all dependents
$ pipenv install
```
**Make sure run all python commands in `pipenv shell` or `pipenv run <command>`**

# instructions
- `utls.py` provides utility functions
- `playground.ipynb` is for test those utils before we actually use them in main file 


# procedure
+ copy pixel with row and col as STI -> tell the wipe and direction of wipe
+ histogram -> test wipe automatically
            -> test resolve
+ IBM



# Tricks

difference between np.matrix np.darray

#### RuntimeError for using matpotlib on Mac
```bash
RuntimeError: Python is not installed as a framework. 
The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. 
See the Python documentation for more information on installing Python as a framework on Mac OS X. 
Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. 
See 'Working with Matplotlib on OSX' inthe Matplotlib FAQ for more information.
```
- Solution:
    ```python
  import matplotlib as mpl
  mpl.use("TkAgg")
  import matplotlib.pyplot as plt
    ```
- [github_issue](https://github.com/MTG/sms-tools/issues/36)

#### Conflict between matplotlib && cv2.imshow
The RuntimeError above occur even if solution applied. 
Try to use only one of them (matplotlib) to show some image.

#### OpenCV store pixels as BGR.

