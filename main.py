from src.model import *
from src.util import utils
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from src.model.detectresult import DetectResult
from src.model.ibm import IbmDetector
from src.model.intersection import IntersectionDetector
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

    def choose_file(self):
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

        self.video_file = filedialog.askopenfilename()
        if not self.video_file:
            return
        self.path_var.set(path_shorten(self.video_file))


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        rows = 0
        while rows < 60:
            self.rowconfigure(rows, weight=1)
            self.columnconfigure(rows, weight=1)
            rows += 1


class LabelImageCombo(tk.Frame):

    IMAGE_WIDTH = 150
    IMAGE_HEIGHT = 150

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.text = tk.StringVar()
        # self.text.set("Transition starts at frame number 1")
        self.text_label = tk.Label(self, textvariable=self.text)
        self.image_label = tk.Label(self)

        self.text_label.pack()
        self.image_label.pack()

    def set_label(self, text):
        self.text.set(text)

    def set_image(self, image_array):
        image = Image.fromarray(image_array, "RGB")
        self.image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.image)
        # self.image_label.image = self.image
        pass


class HistogramDifferenceResultFrame(tk.Frame):
    def detect(self, detector, video_path, threshold):
        detector.set_video(video_path)
        detector.set_threshold(threshold)
        detect_result = detector.detect()
        # TODO: update result in tk
        # Summary
        self.result_text.set("{} | {}".format(detect_result.transition_type, detect_result.transition_direction))

        # start
        self.start_frame.set_image(detect_result.start_frame_image)
        self.start_frame.set_label("Transition Starts at {}th frame".format(detect_result.start_frame_no))
        # middle
        self.middle_frame.set_image(detect_result.middle_frame_image)
        self.middle_frame.set_label("Transition middle at {}th frame".format((detect_result.start_frame_no + detect_result.end_frame_no)//2))
        # end
        self.end_frame.set_image(detect_result.end_frame_image)
        self.end_frame.set_label("Transition ends at {}th frame".format(detect_result.end_frame_no))

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.result_text = tk.StringVar()
        self.result_header = tk.Label(self, textvariable=self.result_text,
                                      font="bold")

        self.start_frame = LabelImageCombo(self)
        self.middle_frame = LabelImageCombo(self)
        self.end_frame = LabelImageCombo(self)

        self.result_header.pack()
        self.start_frame.pack()
        self.middle_frame.pack()
        self.end_frame.pack()


class HistogramDifferencePage(Page):

    def detect(self, detector):
        # accumulate parameters and send to sub detectors

        video_path = self.file_chooser.video_file
        if not video_path:
            messagebox.showinfo("Warning", "Please select a video file to detect.")
            return
        print(video_path)
        threshold = self.threshold_slider.get()
        self.result_frame.detect(detector, video_path, threshold)

    def intersection_detect(self):
        # accumulate parameters and send to sub detectors
        self.detect(self.intersection_detector)
    def ibm_detect(self):
        # accumulate parameters and send to sub detectors
        self.detect(self.ibm_detector)

    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        # init detector
        self.ibm_detector = IbmDetector()
        self.intersection_detector = IntersectionDetector()

        # file chooser
        self.file_chooser = FileChooser()
        file_chooser_label = tk.Label(self, text="Video file")
        file_chooser_button = tk.Button(self, text='Select video file',
                  width=8, height=1,
                  command=self.file_chooser.choose_file)
        file_chooser_path_label = tk.Label(self, textvariable=self.file_chooser.path_var,
                 width=8, height=2)

        file_chooser_label.grid(row=1, column=0, columnspan=4, padx=(30, 10), pady=(40, 20))
        file_chooser_button.grid(row=1, column=4, rowspan=1, columnspan=8,
                                 sticky='NESW', padx=(10, 10), pady=(40, 20))
        file_chooser_path_label.grid(row=1, column=12, columnspan=4, padx=(10, 10), pady=(40, 20))

        # threshold setter
        threshold_label = tk.Label(self, text="Threshold")
        self.threshold_slider = tk.Scale(self, from_=0, to=1, resolution=0.1, orient='horizontal')
        self.threshold_slider.set(0.5)

        threshold_label.grid(row=2, column=0, columnspan=4)
        self.threshold_slider.grid(row=2, column=4, rowspan=1, columnspan=8, pady=(20, 20))

        # two button for two mode
        # TODO: command
        ibm_button = tk.Button(self, text="IBM", command=self.ibm_detect)
        # TODO: command
        intersection_button = tk.Button(self, text="Intersection", command=self.intersection_detect)

        ibm_button.grid(row=3, column=0, columnspan=4, pady=(20, 20))
        intersection_button.grid(row=3, column=4, columnspan=4)

        # result
        self.result_frame = HistogramDifferenceResultFrame(self)
        self.result_frame.grid(row=0, column=16, rowspan=8, columnspan=8)


class CopyPixelPage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

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

        tk.Button(self, text='Select Video File',
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
        copypixel_sti_label.grid(row=19, column=20, rowspan=20, columnspan=20, sticky="NESW", padx=0)

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
    root.geometry('600x700')

    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
