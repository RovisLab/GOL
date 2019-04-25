import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import matplotlib.backends.tkagg as tkagg
import pygmo as pg
from matplotlib.backends.backend_agg import FigureCanvasAgg


class ParetoPlotter(object):
    epoch_list = list()
    optimal_models_epochs = list()

    @staticmethod
    def plot_solution(canvas, solution, accuracy, epoch, epoch_nr):
        print("Accuracy:{0}".format(accuracy))
        print("Solution:{0}".format(solution))
        ParetoPlotter.optimal_models_epochs = list()
        ParetoPlotter.epoch_list.append([accuracy[-1], solution[-1], epoch_nr])
        ParetoPlotter.epoch_list.sort(key=lambda x: x[1], reverse=True)
        ParetoPlotter.epoch_list.sort(key=lambda x: x[0], reverse=True)
        pareto_front_check = [False] * len(accuracy)
        max_solution = ParetoPlotter.epoch_list[0][1]
        for i in range(0, len(pareto_front_check)):
            if ParetoPlotter.epoch_list[i][1] >= max_solution:
                ParetoPlotter.optimal_models_epochs.append(ParetoPlotter.epoch_list[i][2])
                max_solution = ParetoPlotter.epoch_list[i][1]
        print("Epochs list: %s" % ParetoPlotter.epoch_list)
        print("Optimal epochs: %s" % ParetoPlotter.optimal_models_epochs)
        ep_v = [i for i in range(0, epoch)]
        plot_fig = plt.Figure(figsize=(12, 5))
        ax = plot_fig.add_subplot(1, 3, 1)
        ax.plot(solution)

        # ax.set_title("Generalization Energy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy")
        ax.set_xlim([0, 300])
        ax.set_ylim([0, 100])
        ax2 = plot_fig.add_subplot(1, 3, 2)
        # ax2 = plot_fig.add_subplot(2, 1, 2)
        ax2.plot(accuracy)
        # ax2.set_title("Classifier accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlim([0, 300])
        ax2.set_ylim([0, 1.3])

        # Transform the points so that the fronts will be relevant
        plot_points = list()
        for i in range(0, len(accuracy)):
            plot_points.append([1 - accuracy[i], 50 - solution[i]])

        # ---------------------------------- plot pareto front -------------------------------------------------------------

        if len(solution) > 1:

            # Sort the set of points by optimality to get Pareto fronts
            comp = [0, 1]
            fronts, _, _, _ = pg.fast_non_dominated_sorting(plot_points)

            # We define the colors of the fronts (grayscale from black to white)
            cl = list(zip(np.linspace(0.1, 0.9, len(fronts)),
                          np.linspace(0.1, 0.9, len(fronts)),
                          np.linspace(0.1, 0.9, len(fronts))))

            ax3 = plot_fig.add_subplot(1, 3, 3)
            ax3.set_xlabel("Accuracy")
            ax3.set_ylabel("Energy")

            for ndr, front in enumerate(fronts):
                # We plot the points
                for idx in front:
                    ax3.plot(plot_points[idx][comp[0]], plot_points[idx][
                        comp[1]], marker='o', color=cl[ndr])
                # We plot the fronts
                # Frist compute the points coordinates
                x = [plot_points[idx][0] for idx in front]
                y = [plot_points[idx][1] for idx in front]
                # Then sort them by the first objective
                tmp = [(a, b) for a, b in zip(x, y)]
                tmp = sorted(tmp, key=lambda k: k[0])
                # Now plot using step
                ax3.step([c[0] for c in tmp], [c[1]
                                              for c in tmp], color=cl[ndr], where='post')
            plt.show()
        # ------------------------------------------------------------------------------------------------------------------
        return ParetoPlotter.draw_figure(canvas, plot_fig)

    @staticmethod
    def draw_figure(canvas, figure, loc=(0, 0)):
        """ Draw a matplotlib figure onto a Tk canvas
            loc: location of top-left corner of figure on canvas in pixels.
            Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
        """
        figure_canvas_agg = FigureCanvasAgg(figure)
        figure_canvas_agg.draw()
        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
        figure_w, figure_h = int(figure_w), int(figure_h)
        photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

        # Position: convert from top-left anchor to center anchor
        canvas.create_image(loc[0] + figure_w / 2, loc[1] + figure_h / 2, image=photo)

        # Unfortunately, there's no accessor for the pointer to the native renderer
        tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

        # button1 = tk.Button(self, text="Quit", command=self.quit, anchor=W)
        # button1.configure(width=10, activebackground="#33B5E5", relief=FLAT)
        # button1_window = canvas1.create_window(10, 10, anchor=NW, window=button1)

        # Return a handle which contains a reference to the photo object
        # which must be kept live or else the picture disappears
        return photo


# if __name__ == "__main__":
#     w, h = 300, 200
#     window = tk.Tk()
#     window.title("A figure in a canvas")
#     c = tk.Canvas(window, width=w, height=h)
#     c.pack()
#
#     # Generate some example data
#     X = np.linspace(0, 2 * np.pi, 50)
#     Y = np.sin(X)
#
#     # Create the figure we desire to add to an existing canvas
#     fig = plt.Figure(figsize=(2, 1))
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.plot(X, Y)
#
#     # Keep this handle alive, or else figure will disappear
#     fig_x, fig_y = 0, 0
#     fig_photo = draw_figure(c, fig, loc=(fig_x, fig_y))
#     fig_w, fig_h = fig_photo.width(), fig_photo.height()
#
#     # Let Tk take over
#     tk.mainloop()
