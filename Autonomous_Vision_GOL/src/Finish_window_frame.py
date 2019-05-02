from tkinter import Frame, Label
from tkinter import TOP, LEFT, W, CENTER


# Class handles last window frame, informing user when the generation is done
class FinishWindow(Frame):
    def __init__(self, master=None, name=None):
        Frame.__init__(self, master=master, name=name)

        self._master = master
        self._name = name
        self._body_pack_checker = False   # Variable keeps the track regarding whether the body() was called before or not

        width = 300
        height = 100
        x_offset = int((self._master.winfo_screenwidth() - width) / 2)
        y_offset = int((self._master.winfo_screenheight() - height - 50) / 2)
        self._geometry = "{:d}x{:d}+{:d}+{:d}".format(width, height, x_offset, y_offset)

        self.frame_width = width - 50
        self.frame_height = height / 2

    # Function sets geometry of master
    def set_geometry(self):
        self._master.geometry(self._geometry)

    # Function sets the body of the window
    def body(self):
        info_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        info_frame.pack_propagate(False)
        Label(info_frame, text="Generation done!", anchor=W).pack(side=LEFT, anchor=CENTER)
        info_frame.pack(side=TOP)

        self._body_pack_checker = True

    # Function verifies if body() was called before and pack up the frame
    def body_pack(self):
        self.set_geometry()

        if self._body_pack_checker is False:
            self.body()

        self.pack(side=TOP)
