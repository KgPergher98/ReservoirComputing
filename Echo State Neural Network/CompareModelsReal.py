import pandas
import numpy
import matplotlib.pyplot as plt

def changeX(x, tick = 5):
    x = x.tolist()
    aux = []
    for i in x:
        aux.append(i - tick)
    return(aux)

model1 = pandas.read_csv("ParametrosValidacaoReal_Dogecoin.txt")

case1 = model1[model1.WINDOW == 20.0]
#case2 = model2[model2.WINDOW == 1.0]
#case3 = model3[model3.WINDOW == 1.0]

data1 = []
#data2 = []
#data3 = []
ref1 = []
#ref2 = []
#ref3 = []

target = "MAPE"
title_x = "Dogecoin"
if target == "RMSE":
    title_y = "Root Mean Squared Error (RMSE)"
    file_name = "BoxPlot_Real_RMSE_" + title_x + ".png"
    naive_limit = 0.82
elif target == "MAPE":
    title_y = "Mean Absolute Percentage Error (MAPE)"
    file_name = "BoxPlot_Real_MAPE_" + title_x + ".png"
    naive_limit = 7.50
elif target == "HITS":
    title_y = "Hits"
    file_name = "BoxPlot_Real_HITS_" + title_x + ".png"
    naive_limit = 0.500710209338148

for n in case1.NEURONS.unique():
    aux1 = case1[case1.NEURONS == n][target].tolist()
    data1.append(aux1)
    ref1.append(numpy.median(aux1))

#for n in case2.NEURONS.unique():
#    aux2 = case2[case2.NEURONS == n][target].tolist()
#    data2.append(aux2)
#    ref2.append(numpy.median(aux2))

#for n in case2.NEURONS.unique():
#    aux3 = case3[case3.NEURONS == n][target].tolist()
#    data3.append(aux3)
#    ref3.append(numpy.median(aux3))

#print(data1)

tcontrol = 1.75

plt.figure(figsize=(40, 20))
#plt.scatter(case1.NEURONS.unique(), ref1, label = "Modelo I", color = "red")
#plt.scatter(changeX(case2.NEURONS.unique()), ref2, label = "Modelo II", color = "blue")
plt.scatter(case1.NEURONS.unique(), ref1, s = 100, label = "ESN Predictor", color = "blue")
#plt.scatter(changeX(case2.NEURONS.unique()), ref2, label = "Validação", color = "blue")
#plt.scatter(changeX(case3.NEURONS.unique(), tick = 10), ref3, label = "Modelo III", color = "green")
plt.boxplot(data1, positions=case1.NEURONS.unique(), widths = 4, patch_artist = True,
                showmeans = False, showfliers=False,
                medianprops={"color": "black", "linewidth": tcontrol},
                boxprops={"facecolor": "blue", "edgecolor": "black",
                          "linewidth": tcontrol},
                whiskerprops={"color": "black", "linewidth": tcontrol},
                capprops={"color": "black", "linewidth": tcontrol})
#plt.boxplot(data2, positions=changeX(case2.NEURONS.unique()), widths = 4, patch_artist = True,
#                showmeans = False, showfliers=False,
#                medianprops={"color": "black", "linewidth": tcontrol},
#                boxprops={"facecolor": "blue", "edgecolor": "black",
#                          "linewidth": tcontrol},
#                whiskerprops={"color": "black", "linewidth": tcontrol},
#                capprops={"color": "black", "linewidth": tcontrol})
#plt.boxplot(data3, positions=changeX(case3.NEURONS.unique(), tick = 10), widths = 4, patch_artist = True,
#                showmeans = False, showfliers=False,
#                medianprops={"color": "black", "linewidth": tcontrol},
#                boxprops={"facecolor": "green", "edgecolor": "black",
#                          "linewidth": tcontrol},
#                whiskerprops={"color": "black", "linewidth": tcontrol},
#                capprops={"color": "black", "linewidth": tcontrol})
plt.axhline(y=naive_limit, color='r', linestyle='-', label = "Naive Predictor")
plt.legend(prop={'size': 30})
plt.ylabel(title_y, fontsize=32)
plt.xlabel("Tamanho da Camada Oculta/ Reservatório (N)", fontsize=32)
plt.xticks(numpy.int32(case1.NEURONS.unique()), numpy.int32(case1.NEURONS.unique()), fontsize=20)
plt.title(title_x + " (" + target + ")", fontsize=40)
plt.yscale("log")
aux_scale = numpy.linspace(numpy.min(data1), numpy.max([numpy.max(data1), naive_limit]), 10).round(2)
aux_scale = list(aux_scale)
new_scale = [naive_limit]
for sc in aux_scale:
    if abs(sc - naive_limit) > 0.05:
        new_scale.append(sc)
plt.minorticks_off()
plt.yticks(new_scale, new_scale, fontsize=20)
plt.savefig(file_name, bbox_inches='tight')
#plt.show()