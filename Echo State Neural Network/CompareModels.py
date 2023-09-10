import pandas
import numpy
import matplotlib.pyplot as plt

def changeX(x, tick = 5):
    x = x.tolist()
    aux = []
    for i in x:
        aux.append(i - tick)
    return(aux)

model1 = pandas.read_csv("ParametrosTeste_ModeloILorenz.txt")
model2 = pandas.read_csv("ParametrosTeste_ModeloIILorenz.txt")
model3 = pandas.read_csv("ParametrosTeste_ModeloIIILorenz.txt")

case1 = model1[model1.WINDOW == 1.0]
case2 = model2[model2.WINDOW == 1.0]
case3 = model3[model3.WINDOW == 1.0]

data1 = []
data2 = []
data3 = []
ref1 = []
ref2 = []
ref3 = []

title_x = "Atrator de Lorenz (Eixo Z)"

title_y = "Root Mean Squared Error (RMSE)"
target = "RMSE(Z)"
file_name = "BoxPlot_RMSE_Z_Lorenz.png"

#target = "MAPE(X)"
#title_y = "Mean Absolute Percentage Error (MAPE)"
#file_name = "BoxPlot_MAPE_X_Lorenz.png"

for n in case1.NEURONS.unique():
    aux1 = case1[case1.NEURONS == n][target].tolist()
    data1.append(aux1)
    ref1.append(numpy.median(aux1))

for n in case2.NEURONS.unique():
    aux2 = case2[case2.NEURONS == n][target].tolist()
    data2.append(aux2)
    ref2.append(numpy.median(aux2))

for n in case2.NEURONS.unique():
    aux3 = case3[case3.NEURONS == n][target].tolist()
    data3.append(aux3)
    ref3.append(numpy.median(aux3))

#print(data1)

tcontrol = 1.75

plt.figure(figsize=(40,20))
plt.scatter(case1.NEURONS.unique(), ref1, s = 100, label = "Método I", color = "red")
plt.scatter(changeX(case2.NEURONS.unique()), ref2, s = 100, label = "Método II", color = "blue")
plt.scatter(changeX(case3.NEURONS.unique(), tick = 10), ref3, s = 100, label = "Método III", color = "green")
#plt.scatter(case1.NEURONS.unique(), ref1, label = "Treino", color = "red")
#plt.scatter(changeX(case2.NEURONS.unique()), ref2, label = "Validação", color = "blue")

plt.boxplot(data1, positions=case1.NEURONS.unique(), widths = 4, patch_artist = True,
                showmeans = False, showfliers=False,
                medianprops={"color": "black", "linewidth": tcontrol},
                boxprops={"facecolor": "red", "edgecolor": "black",
                          "linewidth": tcontrol},
                whiskerprops={"color": "black", "linewidth": tcontrol},
                capprops={"color": "black", "linewidth": tcontrol})
plt.boxplot(data2, positions=changeX(case2.NEURONS.unique()), widths = 4, patch_artist = True,
                showmeans = False, showfliers=False,
                medianprops={"color": "black", "linewidth": tcontrol},
                boxprops={"facecolor": "blue", "edgecolor": "black",
                          "linewidth": tcontrol},
                whiskerprops={"color": "black", "linewidth": tcontrol},
                capprops={"color": "black", "linewidth": tcontrol})
plt.boxplot(data3, positions=changeX(case3.NEURONS.unique(), tick = 10), widths = 4, patch_artist = True,
                showmeans = False, showfliers=False,
                medianprops={"color": "black", "linewidth": tcontrol},
                boxprops={"facecolor": "green", "edgecolor": "black",
                          "linewidth": tcontrol},
                whiskerprops={"color": "black", "linewidth": tcontrol},
                capprops={"color": "black", "linewidth": tcontrol})
plt.legend(prop={'size': 30})
plt.ylabel(title_y, fontsize=32)
plt.xlabel("Tamanho da Camada Oculta/ Reservatório (N)", fontsize=32)
plt.yticks(fontsize=30)
plt.xticks(numpy.int32(case1.NEURONS.unique()), numpy.int32(case1.NEURONS.unique()), fontsize=20)
plt.title(title_x, fontsize=50)
plt.savefig(file_name, bbox_inches='tight', dpi = 90)