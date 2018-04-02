import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
    def hide(self):
        self.lower()

class TransitionDetectResultPage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        # tk.Label(self, text="Start: 31").pack()
        # back_button = tk.Button(self, text="Back", command=self.lower)
        # back_button.pack()

        tk.Label(self, text="Start: 31").grid(row=0, sticky='W')
        # keep a reference to avoid image loss
        self.start_frame_image = ImageTk.PhotoImage(file="screenshot/ibm/lwipe_chro_IBM.png")
        start_frame = tk.Label(self, image=self.start_frame_image)
        start_frame.grid(row=1)

        back_button = tk.Button(self,text="Back", command=self.lower)
        back_button.grid(row=2)

class TransitionDetectPage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        tk.Label(self, text="Video File").grid(row=0, sticky='W')
        tk.Label(self, text="Threshold").grid(row=1, sticky='W')

        e1 = tk.Entry(self)
        e2 = tk.Entry(self)

        e1.grid(row=0, column=1, columnspan=2)
        e2.grid(row=1, column=1, columnspan=1)

        ibm_button = tk.Button(self, text='IBM', command=self.lower)
        intersection_button = tk.Button(self, text='Intersection')

        ibm_button.grid(row=2, column=0)
        intersection_button.grid(row=2, column=1)


class TransitionTab(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = TransitionDetectPage(self)
        p2 = TransitionDetectResultPage(self)

        p1.place(in_=self, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=self, x=0, y=0, relwidth=1, relheight=1)

        p2.lower()



class Body(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        # Defines and places the notebook widget
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True)
        # nb.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')
        # Adds tab 1 of the notebook
        transition_page = TransitionTab(nb)
        nb.add(transition_page, text='Transition')
        # Note that the column number defaults to 0 if not given.
        sti_page = ttk.Frame(nb)
        nb.add(sti_page, text='STI')

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        rows = 0
        while rows < 50:
            self.rowconfigure(rows, weight=1)
            self.columnconfigure(rows, weight=1)
            rows += 1

        body = Body(self)
        body.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')



if __name__ == "__main__":
    root = tk.Tk()
    root.title('Video Transition Detector')
    root.geometry('500x500')

    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()



# from tkinter import ttk
#
# class MainView(tk.Frame):
#     def __init__(self, *args, **kwargs):
#         tk.Frame
# main = tk.Tk()
# main.title('Notebook Demo')
# main.geometry('500x500')
#
# # gives weight to the cells in the grid
# rows = 0
# while rows < 50:
#     main.rowconfigure(rows, weight=1)
#     main.columnconfigure(rows, weight=1)
#     rows += 1
#
# # text box
#
#
#
# tk.Label(transition_page, text="Video File").grid(row=0, sticky='W')
# tk.Label(transition_page, text="Threshold").grid(row=1, sticky='W')
#
# e1 = tk.Entry(transition_page)
# e2 = tk.Entry(transition_page)
#
# e1.grid(row=0, column=1, columnspan=2)
# e2.grid(row=1, column=1, columnspan=1)
#
# ibm_button = tk.Button(transition_page, text='IBM')
# intersection_button = tk.Button(transition_page, text='Intersection')
#
# ibm_button.grid(row=2, column=0)
# intersection_button.grid(row=2, column=1)
#
# # Adds tab 2 of the notebook
#
# main.mainloop()