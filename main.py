import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)


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
