from src.model import *
from src.util import utils
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
import math
import matplotlib
from PIL import Image, ImageTk

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# global variable
video_file = None
to_chromatic = True
threshold = 0.5
result_message = None
detect_mode = None
sample_mode = None

# tkinter
window = tk.Tk()
window.title('Video Transition Detector')
window.geometry('800x800')

# gives weight to the cells in the grid
rows = 0
while rows < 80:
    window.rowconfigure(rows, weight=1)
    window.columnconfigure(rows, weight=1)
    rows += 1

# Introduction Label
str = " Author: Bowen Chen | Haipeng Li"
label_introduction = tk.Label(window,
                              text=str,
                              # bg='green',
                              font=('Arial', 12),
                              width=50, height=5
                              )
label_introduction.grid(row=5, column=0, rowspan=10, columnspan=80, sticky='NESW')  # 固定窗口位置

# NoteBook container
# Defines and places the notebook widget
nb_container = ttk.Notebook(window)
nb_container.grid(row=20, column=0, rowspan=60, columnspan=80, sticky='NESW')

# Adds tab 1 of the notebook
histogram_page = ttk.Frame(nb_container)
nb_container.add(histogram_page, text='Histogram Difference')

# Adds tab 2 of the notebook
copypixel_page = ttk.Frame(nb_container)
nb_container.add(copypixel_page, text='Copy Pixel')

##### Copypixel Page
# gives weight to the cells in the grid
rows = 0
while rows < 60:
    copypixel_page.rowconfigure(rows, weight=1)
    copypixel_page.columnconfigure(rows, weight=1)
    rows += 1

# File Selector
path_var = tk.StringVar()


def select_file():
    global video_file
    video_file = filedialog.askopenfilename()

    def path_shorten(path):
        while len(path) > 40:
            i = 0
            for i in range(len(path)):
                if path[i] == '/':
                    path = path[i + 1:]
                    break
            if i == len(path) - 1:
                return path
        return path

    path_var.set("SOMEPATH/" + path_shorten(video_file))
    print(video_file)


tk.Button(copypixel_page, text='Slect Video File',
          width=4, height=1,
          command=select_file).grid(row=5, column=0, rowspan=1, columnspan=4, sticky='NESW')

# File Selected Text
tk.Label(copypixel_page, textvariable=path_var,
         width=40, height=2).grid(row=5, column=4, rowspan=1, columnspan=40, sticky='NESW')

# Sampling Choice
var = tk.StringVar()


def samling_mode_set():
    global sample_mode
    sample_mode = var.get()


tk.Label(copypixel_page,
         text="Sampling Choice",
         # bg='green',
         font=('Arial', 18),
         width=3, height=1
         ).grid(row=19, column=0, rowspan=1, columnspan=4, sticky='NESW')

row_choice = tk.Radiobutton(copypixel_page, text='MidRow Sampling',
                            variable=var, value='row',
                            command=samling_mode_set)
row_choice.grid(row=20, column=0, rowspan=3, columnspan=4, sticky="NESW", padx=0)
col_choice = tk.Radiobutton(copypixel_page, text='MidCol Sampling',
                            variable=var, value='col',
                            command=samling_mode_set)
col_choice.grid(row=23, column=0, rowspan=3, columnspan=4, sticky="NESW", padx=0)

# label to show sti image
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 180
loading_img = cv2.imread(filename='../../image/loading.jpg', flags=cv2.IMREAD_COLOR)
loading_img = cv2.resize(loading_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
loading_img = Image.fromarray(loading_img, "RGB")
loading_img = ImageTk.PhotoImage(loading_img)

# loading_img = ImageTk.PhotoImage(file='../../image/loading.gif')
copypixel_sti_label = tk.Label(copypixel_page)
copypixel_sti_label.grid(row=19, column=10, rowspan=20, columnspan=30, sticky="NESW", padx=0)


# Start detection
def detection():
    if video_file == None:
        # warning dialog
        messagebox.showinfo("Warning", "Please select a video file to detect.")
        return
    if sample_mode == None:
        # warning dialog
        messagebox.showinfo("Warning", "Please select a sampling mode.")
        return
    # set loading image
    copypixel_sti_label.config(image=loading_img)
    copypixel_sti_label.image = loading_img
    # pixel sampling
    if sample_mode == "row":
        sti = utils.pixel_copy_sti(video_file, use_row=True)
    else:
        sti = utils.pixel_copy_sti(video_file, use_row=False)
    # set sti image
    sti_img = cv2.resize(sti, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    sti_img = Image.fromarray(sti_img, "RGB")
    sti_img = ImageTk.PhotoImage(sti_img)
    copypixel_sti_label.config(image=sti_img)
    copypixel_sti_label.image = sti_img


tk.Button(copypixel_page,
          text="Start Detection",
          width=10, height=2, command=detection).grid(row=28, column=0, rowspan=1, columnspan=4, sticky="NESW")
##### END of Copypixel Page






window.mainloop()
