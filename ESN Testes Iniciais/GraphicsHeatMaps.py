from inspect import stack
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid

class Analysis():

    def __init__(self, file = "ParametrosScreenLorenz.txt",
                 file_name = "LorenzAttractor", file_title = "Lorenz Attractor") -> None:

        self.data_mor = pandas.read_csv("C:\\Users\\kgper\\OneDrive\\Área de Trabalho\\Neural Networks Codes\\ReservoirComputing\\" + file)
        self.data_mor.drop(["TRAIN", "NEURONS"], axis = 1, inplace = True)

        self.set_radius = self.data_mor.RADIUS.unique()
        self.set_leaking = self.data_mor.LEAKING.unique()
        self.set_sparsity = self.data_mor.SPARSITY.unique()
        self.set_scale = self.data_mor.SCALE.unique()

        self.file_name = file_name
        self.file_title = file_title

    def call_set(self, calling = "RADIUS"):

        if calling == "RADIUS":
            return self.set_radius
        elif calling == "LEAKING":
            return self.set_leaking
        elif calling == "SPARSITY":
            return self.set_sparsity
        elif calling == "SCALE":
            return self.set_scale

    def create_median(self, var_x = "RADIUS", var_y = "LEAKING", case = "MAPE"):

        x = Analysis.call_set(self, calling = var_x)
        y = Analysis.call_set(self, calling = var_y)
        var = pandas.DataFrame(0, index = x, columns = y)

        for x_data in x:
            if var_x == "RADIUS":
                d = self.data_mor[self.data_mor.RADIUS == x_data]
            elif var_x == "LEAKING":
                d = self.data_mor[self.data_mor.LEAKING == x_data]
            elif var_x == "SPARSITY":
                d = self.data_mor[self.data_mor.SPARSITY == x_data]
            elif var_x == "SCALE":
                d = self.data_mor[self.data_mor.SCALE == x_data]
            for y_data in y:
                if var_y == "LEAKING":
                    k = d[d.LEAKING == y_data]
                elif var_y == "RADIUS":
                    k = d[d.RADIUS == y_data]
                elif var_y == "SCALE":
                    k = d[d.SCALE == y_data]
                elif var_y == "SPARSITY":
                    k = d[d.SPARSITY == y_data]
                if case == "MAPE":
                    var.loc[x_data, y_data] = numpy.median(k.MAPE)
                elif case == "MSE":
                    var.loc[x_data, y_data] = numpy.median(k.MSE)
        return x, y, var

    def images(self, var_x, var_y):

        y, x, var = analysis.create_median(var_x = var_x, var_y = var_y, case = "MAPE")

        fig, ax = plt.subplots(figsize=(10,10))
        sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', interpolation='gaussian')
        plt.colorbar(sh)
        ax.set_yticks(ticks = range(len(y)), labels = y)
        ax.set_xticks(ticks = range(len(x)), labels = x)
        ax.set_title("MAPE (logscale) - " + self.file_title + ": " + var_y + " (X) x " + var_x + " (Y)")
        plt.savefig(self.file_name + "_MAPE_HeatMap_" + var_x + "_" + var_y + ".png")
        plt.close()

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #x, y = numpy.meshgrid(x, y)
        #surf = ax.plot_surface(x, y, var, cmap=cm.inferno,
        #                    antialiased=False)
        #plt.savefig(self.file_name + "_Surface_" + var_x + "_" + var_y + ".png")
        #plt.close()

files = ["ParametrosScreenLorenz.txt", "ParametrosScreenHenon.txt", 
         "ParametrosScreenMackey17.txt", "ParametrosScreenMackey22.txt"]
file_names = ["LorenzAttractor", "HenonMap", "Mackey17", "Mackey22"]
file_titles = ["Lorenz Attractor", "Henon Map", "Mackey Glass 17", "Mackey Glass 22"]

for i in range(len(files)):
    analysis = Analysis(file = files[i], file_name = file_names[i], file_title = file_titles[i])
    analysis.images(var_x = "RADIUS", var_y = "LEAKING")
    analysis.images(var_x = "RADIUS", var_y = "SPARSITY")
    analysis.images(var_x = "RADIUS", var_y = "SCALE")
    analysis.images(var_x = "LEAKING", var_y = "SCALE")
    analysis.images(var_x = "LEAKING", var_y = "SPARSITY")
    analysis.images(var_x = "SCALE", var_y = "SPARSITY")