import pandas
import numpy
import matplotlib.pyplot as plt

def changeX(x, tick = 5):
    x = x.tolist()
    aux = []
    for i in x:
        aux.append(i - tick)
    return(aux)

model1 = pandas.read_csv("ParametrosLorenzTesteModeloI.txt")
model2 = pandas.read_csv("ParametrosLorenzTesteModeloII.txt")
model3 = pandas.read_csv("ParametrosLorenzTesteModeloIII.txt")

case1 = model1[model1.TRAIN == 10]
case2 = model2[model2.TRAIN == 10]
case3 = model3[model3.TRAIN == 10]

data1 = []
data2 = []
data3 = []
ref1 = []
ref2 = []
ref3 = []

target = "MSE(X)"

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

plt.figure(figsize=(25,15))
plt.scatter(case1.NEURONS.unique(), ref1, label = "Modelo I", color = "red")
plt.scatter(changeX(case2.NEURONS.unique()), ref2, label = "Modelo II", color = "blue")
plt.scatter(changeX(case3.NEURONS.unique(), tick = 10), ref3, label = "Modelo III", color = "green")
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
plt.legend()
plt.ylabel("Mean Squared Error (MSE)")
plt.xlabel("Tamanho da camada oculta (neurônios)")
plt.xticks(numpy.int32(case1.NEURONS.unique()), numpy.int32(case1.NEURONS.unique()))
plt.title("Atrator de Lorenz (eixo X)")
plt.savefig("bplot_lorenz_mse_x.png", bbox_inches='tight')

"""
plt.figure(figsize=(15,15))
plt.scatter(case1.NEURONS, case1["MSE(Z)"], label = "Modelo I", color = "black")
plt.scatter(changeX(case2.NEURONS), case2["MSE(Z)"], label = "Modelo II", color = "blue")
plt.scatter(changeX(case3.NEURONS, tick = 10), case3["MSE(Z)"], label = "Modelo III", color = "red")
plt.legend()
plt.ylabel("Mean Squared Error (MSE)")
plt.xlabel("Tamanho da camada oculta (neurônios)")
plt.title("Atrator de Lorenz (eixo Z)")
plt.show()
"""