from tkinter import Frame, LabelFrame, Canvas
from tkinter import TOP, NW, BOTH
from PIL import Image, ImageTk
from Filters_applier import FiltersApplier


# Class handles parameters sampling frame
class ApplyFiltersFrame(Frame):
    def __init__(self, master=None, name=None, width=None, height=None):
        Frame.__init__(self, master=master, name=name)

        self._master = master

        # Variable keeps the track regarding whether the body() was called before or not
        self._body_pack_checker = False

        self.dir_path = "../bin/misc"
        self.min_file_name = "/Minimum.png"
        self.max_file_name = "/Maximum.png"

        self.dims = (265, 265)

        self.config(width=width, height=height)
        self.pack_propagate(False)

        self.apply_filters_min_frame = LabelFrame(self, text="Minimum:", width=width, height=height/2-75, bd=0)
        self.apply_filters_min_frame.pack_propagate(False)
        self.minimum_canvas = Canvas(self.apply_filters_min_frame)
        self.min_image_to_plot = ImageTk
        self.min_image = Canvas()

        self.apply_filters_max_frame = LabelFrame(self, text="Maximum:", width=width, height=height/2-75, bd=0)
        self.apply_filters_max_frame.pack_propagate(False)
        self.maximum_canvas = Canvas(self.apply_filters_max_frame)
        self.max_image_to_plot = ImageTk
        self.max_image = Canvas()

    # Function sets the body of the frame
    def body(self, image_path=None):
        pil_image = Image.open(image_path)
        resized_pil_image = pil_image.resize(size=self.dims)

        self.min_image_to_plot = ImageTk.PhotoImage(resized_pil_image)
        self.min_image = Canvas.create_image(self.minimum_canvas, 0, 0, image=self.min_image_to_plot, anchor=NW)
        self.minimum_canvas.pack(fill=BOTH, side=TOP)
        self.apply_filters_min_frame.pack(side=TOP, pady=25)

        self.max_image_to_plot = ImageTk.PhotoImage(resized_pil_image)
        self.max_image = Canvas.create_image(self.maximum_canvas, 0, 0, image=self.max_image_to_plot, anchor=NW)
        self.maximum_canvas.pack(fill=BOTH, side=TOP)
        self.apply_filters_max_frame.pack(side=TOP)

        self._body_pack_checker = True

    # Function verifies if body() was called before and packs up the frame
    def body_pack(self, side, image_path=None):
        if self._body_pack_checker is False:
            self.body(image_path=image_path)

        self.pack(side=side)

    # Function calls the filters applier
    def sampling_filters(self, config_file_path):
        applier = FiltersApplier(config_file_path)
        applier.callback()

        pil_image_min = Image.open(self.dir_path + self.min_file_name)
        if pil_image_min.size[0] > self.dims[0] or pil_image_min.size[1] > self.dims[1]:
            pil_image_min = pil_image_min.resize(size=self.dims)
        self.min_image_to_plot = ImageTk.PhotoImage(pil_image_min)
        self.min_image = Canvas.create_image(self.minimum_canvas, 2, 2, image=self.min_image_to_plot, anchor=NW)

        pil_image_max = Image.open(self.dir_path + self.max_file_name)
        if pil_image_max.size[0] > self.dims[0] or pil_image_max.size[1] > self.dims[1]:
            pil_image_max = pil_image_max.resize(size=self.dims)
        self.max_image_to_plot = ImageTk.PhotoImage(pil_image_max)
        self.max_image = Canvas.create_image(self.maximum_canvas, 2, 2, image=self.max_image_to_plot, anchor=NW)

    # Function removes no longer useful misc files (its own if any or passed from outside)
    def delete_misc_content(self, other_files=None):
        import os

        if os.path.isfile(self.dir_path + self.min_file_name):
            os.remove(self.dir_path + self.min_file_name)

        if os.path.isfile(self.dir_path + self.max_file_name):
            os.remove(self.dir_path + self.max_file_name)

        files_to_remove = [other_files]
        for it in range(0, len(files_to_remove)):
            if os.path.isfile(files_to_remove[it]):
                os.remove(files_to_remove[it])


if __name__ == "__main__":
    from tkinter import Tk, LEFT

    root = Tk()
    filters_frame = ApplyFiltersFrame(master=root, width=500, height=750)
    filters_frame.body_pack(side=LEFT, image_path="D:/dev/data/ASG/Input/Templates/Danger_signs/Animals.png")
    root.mainloop()
