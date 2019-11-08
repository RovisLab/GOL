from tkinter import Frame, Label
from tkinter import TOP, LEFT, W, CENTER, HORIZONTAL
from tkinter.ttk import Progressbar


# Class handles GUI activity while samples are generated
class GenerationWindow(Frame):
    def __init__(self, master=None, name=None):
        Frame.__init__(self, master=master, name=name)

        self._master = master
        self._name = name

        # Variable keeps the track regarding whether the body() was called before or not
        self._body_pack_checker = False

        width = 300
        height = 100
        x_offset = int((self._master.winfo_screenwidth() - width) / 2)
        y_offset = int((self._master.winfo_screenheight() - height - 50) / 2)
        self._geometry = "{:d}x{:d}+{:d}+{:d}".format(width, height, x_offset, y_offset)

        self.frame_width = width - 50
        self.frame_height = height / 2

        self.progressbar_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.progressbar_frame.pack_propagate(False)
        self.generation_progressbar = Progressbar(self.progressbar_frame, orient=HORIZONTAL,
                                                  length=self.frame_width, mode='indeterminate')

    # Function sets geometry of master
    def set_geometry(self):
        self._master.geometry(self._geometry)

    # Functions sets the body of the window
    def body(self):
        label_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        label_frame.pack_propagate(False)
        Label(label_frame, text="Please wait. Generating...", anchor=W).pack(side=LEFT, anchor=CENTER)
        label_frame.pack(side=TOP)

        self.generation_progressbar.pack(side=LEFT, anchor=CENTER)
        self.generation_progressbar.start()
        self.progressbar_frame.pack(side=TOP)

        self._body_pack_checker = True

    # Function verifies if body() was called before and packs up the frame
    def body_pack(self):
        self.set_geometry()

        if self._body_pack_checker is False:
            self.body()

        self.pack(side=TOP)
