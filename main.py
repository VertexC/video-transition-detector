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



# global variable
video_file = None
sample_mode = None

class FileChooser():
    path_var = None
    video_file = None

    def __init__(self):
        self.path_var = tk.StringVar()

    def get_path_var(self):
        return self.path_var

    def get_video_file(self):
        return self.video_file

    def choose_file(self):
        self.video_file = filedialog.askopenfilename()

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

        self.path_var.set("SOMEPATH/" + path_shorten(self.video_file))
        print(self.video_file)


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)



class HistogramDifferencePage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        # file chooser
        file_chooser = FileChooser()
        tk.Button(self, text='Slect Video File',
                  width=4, height=1,
                  command=file_chooser.choose_file).grid(row=5, column=0, rowspan=1, columnspan=4, sticky='NESW')
        tk.Label(self, textvariable=file_chooser.get_path_var,
                 width=40, height=2).grid(row=5, column=4, rowspan=1, columnspan=40, sticky='NESW')

        # threshold setter
        threshold_label = tk.Label(self, text="Threshold")
        threshold_slider = tk.Scale(self, from_=0, to=1, resolution=0.1, orient='horizontal')
        threshold_slider.set(0.5)

        threshold_slider.grid(row=7, rowspan=3, column=2, columnspan=2)
        threshold_label.grid(row=8, column=0, columnspan=2)

        # two button for two mode

        # result


class CopyPixelPage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        rows = 0
        while rows < 60:
            self.rowconfigure(rows, weight=1)
            self.columnconfigure(rows, weight=1)
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
            # print(video_file)

        tk.Button(self, text='Slect Video File',
                  width=4, height=1,
                  command=select_file).grid(row=5, column=0, rowspan=1, columnspan=4, sticky='NESW')

        # File Selected Text
        tk.Label(self, textvariable=path_var,
                 width=40, height=2).grid(row=5, column=4, rowspan=1, columnspan=40, sticky='NESW')

        # Sampling Choice
        var = tk.StringVar()

        def samling_mode_set():
            global sample_mode
            sample_mode = var.get()

        tk.Label(self,
                 text="Sampling Choice",
                 # bg='green',
                 font=('Arial', 18),
                 width=3, height=1
                 ).grid(row=19, column=0, rowspan=1, columnspan=4, sticky='NESW')

        row_choice = tk.Radiobutton(self, text='MidRow Sampling',
                                    variable=var, value='row',
                                    command=samling_mode_set)
        row_choice.grid(row=20, column=0, rowspan=3, columnspan=4, sticky="NESW", padx=0)
        col_choice = tk.Radiobutton(self, text='MidCol Sampling',
                                    variable=var, value='col',
                                    command=samling_mode_set)
        col_choice.grid(row=23, column=0, rowspan=3, columnspan=4, sticky="NESW", padx=0)

        # label to show sti image
        IMAGE_WIDTH = 200
        IMAGE_HEIGHT = 180
        loading_img = cv2.imread(filename='./image/loading.jpg', flags=cv2.IMREAD_COLOR)
        loading_img = cv2.resize(loading_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        loading_img = Image.fromarray(loading_img, "RGB")
        loading_img = ImageTk.PhotoImage(loading_img)

        copypixel_sti_label = tk.Label(self)
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

        tk.Button(self,
                  text="Start Detection",
                  width=10, height=2, command=detection).grid(row=35, column=0, rowspan=1, columnspan=4, sticky="NESW")


class NoteBook(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        # Defines and places the notebook widget
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True)

        # add Page to Notebook
        copypixel_page = CopyPixelPage(nb)
        nb.add(copypixel_page, text='Copy Pixel')
        histogram_difference_page = HistogramDifferencePage(nb)
        nb.add(histogram_difference_page, text='Histogram Difference')

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        rows = 0
        while rows < 80:
            self.rowconfigure(rows, weight=1)
            self.columnconfigure(rows, weight=1)
            rows += 1


        label_introduction = tk.Label(self,
                                      text=" Author: Bowen Chen | Haipeng Li",
                                      # bg='green',
                                      font=('Arial', 12),
                                      width=50, height=5
                                      ).grid(row=5, column=0, rowspan=10, columnspan=80, sticky='NESW')

        body = NoteBook(self)
        body.grid(row=20, column=0, rowspan=60, columnspan=80, sticky='NESW')



if __name__ == "__main__":
    root = tk.Tk()
    root.title('Video Transition Detector')
    root.geometry('800x800')

    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
