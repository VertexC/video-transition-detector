import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image


class FileChooser():
    path_var = tk.StringVar()
    video_file = None

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

        # file chooser
        file_chooser = FileChooser()
        tk.Button(self, text='Slect Video File',
                  width=4, height=1,
                  command=file_chooser.choose_file).grid(row=5, column=0, rowspan=1, columnspan=4, sticky='NESW')
        tk.Label(self, textvariable=file_chooser.get_path_var,
                 width=40, height=2).grid(row=5, column=4, rowspan=1, columnspan=40, sticky='NESW')

        # threshold setter
        threshold_label = tk.Label(self, text="Threshold")
        threshold_slider = tk.Scale(self, from_=0, to=1, tickinterval=0.1)
        threshold_slider.set(0.5)

        threshold_label.grid(row=7, column=0, columnspan=2, sticky='NESW')
        threshold_slider.grid(row=7, column=2, columnspan=2, sticky='NESW')


        # two button for two mode

        # result


class HistogramDifferencePage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(*args, **kwargs)


class NoteBook(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        # Defines and places the notebook widget
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True)

        # add Page to Notebook


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
