from inspect import stack
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
from matplotlib import cm

class Analysis():

    def __init__(self) -> None:
        self.data_mor = pandas.read_csv("ParametrosTreinoReal_WEGE.txt")
        self.data_mor.drop(["WINDOW", "NEURONS"], axis = 1, inplace = True)
        print(self.data_mor)
        #self.data_mor = self.data_mor.astype(numpy.float64)
        #print(self.data_mor)

        self.set_radius = self.data_mor.RADIUS.unique()
        self.set_leaking = self.data_mor.LEAK.unique()
        self.set_sparsity = self.data_mor.SPARSITY.unique()
        self.set_scale = self.data_mor.SCALE.unique()

    def call_set(self, calling = "RADIUS"):
        if calling == "RADIUS":
            return self.set_radius
        elif calling == "LEAKING":
            return self.set_leaking
        elif calling == "SPARSITY":
            return self.set_sparsity
        elif calling == "SCALE":
            return self.set_scale

    def create_median(self, var_x = "RADIUS", var_y = "LEAKING"):
        norm = 0
        x = Analysis.call_set(self, calling = var_x)
        y = Analysis.call_set(self, calling = var_y)
        var = pandas.DataFrame(0, index = x, columns = y)
        for x_data in x:
            if var_x == "RADIUS":
                d = self.data_mor[self.data_mor.RADIUS == x_data]
            elif var_x == "LEAKING":
                d = self.data_mor[self.data_mor.LEAK == x_data]
            elif var_x == "SPARSITY":
                d = self.data_mor[self.data_mor.SPARSITY == x_data]
            elif var_x == "SCALE":
                d = self.data_mor[self.data_mor.SCALE== x_data]
            for y_data in y:
                if var_y == "LEAKING":
                    k = d[d.LEAK == y_data]
                elif var_y == "RADIUS":
                    k = d[d.RADIUS == y_data]
                elif var_y == "SCALE":
                    k = d[d.SCALE == y_data]
                elif var_y == "SPARSITY":
                    k = d[d.SPARSITY == y_data]
                var.loc[x_data, y_data] = numpy.median(k.loc[:,"MAPE"].astype(float))
        return y, x, var

analysis = Analysis()
file_name = "WEGE"
file_title = "WEGE"

fsize = (12,12)

x, y, var = analysis.create_median(var_x = "RADIUS", var_y = "LEAKING")

fig, ax = plt.subplots(figsize = fsize)
sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', aspect = 'auto', interpolation='gaussian')
plt.colorbar(sh)
ax.set_yticks(ticks = range(len(y)), labels = y)
ax.set_xticks(ticks = range(len(x)), labels = x)
#ax.set_title("MAPE (logscale) - " + file_title + ": Leaking Rate (X) x Spectral Radius (Y)")
plt.ylabel("Spectral Radius")
plt.xlabel("Leaking Rate")
plt.savefig("HeatMap_" + file_name + "_LeakingRadius.png", bbox_inches='tight')
plt.close()

x, y, var = analysis.create_median(var_x = "RADIUS", var_y = "SPARSITY")

fig, ax = plt.subplots(figsize = fsize)
sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', aspect = 'auto', interpolation='gaussian')
plt.colorbar(sh)
ax.set_yticks(ticks = range(len(y)), labels = y)
ax.set_xticks(ticks = range(len(x)), labels = x)
#ax.set_title("MAPE (logscale) - " + file_title + ": Sparsity (X) x Spectral Radius (Y)")
plt.ylabel("Spectral Radius")
plt.xlabel("Sparsity")
plt.savefig("HeatMap_" + file_name + "_SparsityRadius.png", bbox_inches='tight')
plt.close()

x, y, var = analysis.create_median(var_x = "RADIUS", var_y = "SCALE")

fig, ax = plt.subplots(figsize = fsize)
sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', aspect = 'auto', interpolation='gaussian')
plt.colorbar(sh)
ax.set_yticks(ticks = range(len(y)), labels = y)
ax.set_xticks(ticks = range(len(x)), labels = x)
#ax.set_title("MAPE (logscale) - " + file_title + ": Input Scalling (X) x Spectral Radius (Y)")
plt.ylabel("Spectral Radius")
plt.xlabel("Input Scaling")
plt.savefig("HeatMap_" + file_name + "_ScaleRadius.png", bbox_inches='tight')
plt.close()

x, y, var = analysis.create_median(var_x = "LEAKING", var_y = "SCALE")

fig, ax = plt.subplots(figsize = fsize)
sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', aspect = 'auto', interpolation='gaussian')
plt.colorbar(sh)
ax.set_yticks(ticks = range(len(y)), labels = y)
ax.set_xticks(ticks = range(len(x)), labels = x)
#ax.set_title("MAPE (logscale) - " + file_title + ": Input Scalling (X) x Leaking Rate (Y)")
plt.ylabel("Leaking Rate")
plt.xlabel("Input Scaling")
plt.savefig("HeatMap_" + file_name + "_ScaleLeaking.png", bbox_inches='tight')
plt.close()

x, y, var = analysis.create_median(var_x = "LEAKING", var_y = "SPARSITY")

fig, ax = plt.subplots(figsize = fsize)
sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', aspect = 'auto', interpolation='gaussian')
plt.colorbar(sh)
ax.set_yticks(ticks = range(len(y)), labels = y)
ax.set_xticks(ticks = range(len(x)), labels = x)
#ax.set_title("MAPE (logscale) - " + file_title + ": Sparsity (X) x Leaking Rate (Y)")
plt.ylabel("Leaking Rate")
plt.xlabel("Sparsity")
plt.savefig("HeatMap_" + file_name + "_SparsityLeaking.png", bbox_inches='tight')
plt.close()

x, y, var = analysis.create_median(var_x = "SCALE", var_y = "SPARSITY")

fig, ax = plt.subplots(figsize = fsize)
sh = ax.imshow(var.apply(numpy.log10), cmap = 'inferno', aspect = 'auto', interpolation='gaussian')
plt.colorbar(sh)
ax.set_yticks(ticks = range(len(y)), labels = y)
ax.set_xticks(ticks = range(len(x)), labels = x)
#ax.set_title("MAPE (logscale) - " + file_title + ": Sparsity (X) x Input Scalling (Y)")
plt.ylabel("Input Scaling")
plt.xlabel("Sparsity")
plt.savefig("HeatMap_" + file_name + "_ScaleSparsity.png", bbox_inches='tight')


#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#x, y = numpy.meshgrid(x, y)
#surf = ax.plot_surface(x, y, var, cmap=cm.inferno,
#                       antialiased=False)
#plt.show()


"""
data_mor = pandas.read_csv("C:\\Users\\kgper\\OneDrive\\Área de Trabalho\\Neural Networks Codes\\ReservoirComputing\\RodagemLorenz20220724.txt")
data_mor["NMSE"] = numpy.log10(data_mor["NMSE"])
data_mor["MAPE"] = numpy.log10(data_mor["MAPE"])
sufix = "(Lorenz Attractor)"
prefix = "LorenzAttractor"
print(data_mor)

sns.set_theme(style="whitegrid")
sns.boxplot(x="TRAIN", y="NMSE", data=data_mor).set(title="Distribuição NMSE por quantidade de dados de treinamento " + sufix)
plt.savefig(prefix + "TrainNMSE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="NEURONS", y="NMSE", data=data_mor).set(title="Distribuição NMSE por quantidade de neurônios " + sufix)
plt.savefig(prefix + "NeuronsNMSE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="RADIUS", y="NMSE", data=data_mor).set(title="Distribuição NMSE por raio espectral " + sufix)
plt.savefig(prefix + "RadiusNMSE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="LEAKING", y="NMSE", data=data_mor).set(title="Distribuição NMSE por taxa de vazamento " + sufix)
plt.savefig(prefix + "LeakingNMSE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="SPARSITY", y="NMSE", data=data_mor).set(title="Distribuição NMSE por esparsidade " + sufix)
plt.savefig(prefix + "SparsityNMSE.png")
plt.close()

# MAPE

sns.set_theme(style="whitegrid")
sns.boxplot(x="TRAIN", y="MAPE", data=data_mor).set(title="Distribuição MAPE por quantidade de dados de treinamento " + sufix)
plt.savefig(prefix + "TrainMAPE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="NEURONS", y="MAPE", data=data_mor).set(title="Distribuição MAPE por quantidade de neurônios " + sufix)
plt.savefig(prefix + "NeuronsMAPE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="RADIUS", y="MAPE", data=data_mor).set(title="Distribuição MAPE por raio espectral " + sufix)
plt.savefig(prefix + "RadiusMAPE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="LEAKING", y="MAPE", data=data_mor).set(title="Distribuição MAPE por taxa de vazamento " + sufix)
plt.savefig(prefix + "LeakingMAPE.png")
plt.close()

sns.set_theme(style="whitegrid")
sns.boxplot(x="SPARSITY", y="MAPE", data=data_mor).set(title="Distribuição MAPE por esparsidade " + sufix)
plt.savefig(prefix + "SparsityMAPE.png")
plt.close()
"""

"""
for neurons in [50, 100, 150, 200]:
    data = data_mor[data_mor.NEURONS == neurons]
    network = "_" + str(neurons) + "NeuronsHenonMap"


    x_set = data.TRAIN.unique()
    y_set = data.RADIUS.unique()
    mse = pandas.DataFrame(0, index=x_set, columns=y_set)
    nmse = pandas.DataFrame(0, index=x_set, columns=y_set)
    mape = pandas.DataFrame(0, index=x_set, columns=y_set)

    for t in range(data.shape[0]):
        mse.loc[data.iloc[t,0], data.iloc[t,2]] = data.iloc[t,5]
        nmse.loc[data.iloc[t,0], data.iloc[t,2]] = data.iloc[t,6]
        mape.loc[data.iloc[t,0], data.iloc[t,2]] = data.iloc[t,7]

    x, y = x_set, y_set

    fig = go.Figure(data=[go.Surface(z=mse.T, x=x, y=y)])
    fig.update_layout(scene = dict(
                        xaxis_title='TRAINNING SIZE',
                        yaxis_title='SPECTRAL RADIUS',
                        zaxis_title='MSE'),
                    title = 'Mean Squared Error', autosize = False,
                    width = 800, height = 800,
                    margin = dict(l=65, r=50, b=65, t=90))
    fig.write_html("MSE" + network + ".html")

    fig = go.Figure(data=[go.Surface(z=nmse.T, x=x, y=y)])
    fig.update_layout(scene = dict(
                        xaxis_title='TRAINNING SIZE',
                        yaxis_title='SPECTRAL RADIUS',
                        zaxis_title='NMSE'),
                    title = 'Normalized Mean Squared Error', autosize = False,
                    width = 800, height = 800,
                    margin = dict(l=65, r=50, b=65, t=90))
    fig.write_html("NMSE" + network + ".html")

    fig = go.Figure(data=[go.Surface(z=mape.T, x=x, y=y)])
    fig.update_layout(scene = dict(
                        xaxis_title='TRAINNING SIZE',
                        yaxis_title='SPECTRAL RADIUS',
                        zaxis_title='MAPE'),
                    title = 'Mean Absolute Percentage Error', autosize = False,
                    width = 800, height = 800,
                    margin = dict(l=65, r=50, b=65, t=90))
    fig.write_html("MAPE" + network + ".html")
"""