import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

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
var = tk.StringVar()

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

# Copypixel Page
# gives weight to the cells in the grid
rows = 0
while rows < 60:
    copypixel_page.rowconfigure(rows, weight=1)
    copypixel_page.columnconfigure(rows, weight=1)
    rows += 1


# File Selector
def select_file():
    global video_file
    video_file = filedialog.askopenfilename()
    def path_shorten(path):
        while len(path) > 40:
            i = 0
            for i in range(len(path)):
                if path[i] == '/':
                    path = path[i+1:]
                    break
            if i == len(path)-1:
                return path
        return path
    var.set("SOMEPATH/" + path_shorten(video_file))
    print(video_file)

tk.Button(copypixel_page, text='Slect Video File',
          width=4, height=1,
          command=select_file).grid(row=5, column=0, rowspan=1, columnspan=4, sticky='NESW')

# File Selected Text
tk.Label(copypixel_page, textvariable=var,
         width=40, height=2).grid(row=5, column=4, rowspan=1, columnspan=40, sticky='NESW')

# Sampling Choice
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

# start detection

window.mainloop()
