from turtle import width
import pandas
import numpy
import matplotlib.pyplot as plt

data = pandas.read_csv("TesteLorenzGN0_00.txt")
data_gn1 = pandas.read_csv("TesteLorenzGN0_0005.txt")
data_gn2 = pandas.read_csv("TesteLorenzGN0_0010.txt")
data_gn3 = pandas.read_csv("TesteLorenzGN0_0015.txt")
data_gn4 = pandas.read_csv("TesteLorenzGN0_0020.txt")

target = "MSE(Z)"

plt.figure(figsize = (10,10))
plt.title("Lorenz")
plt.xlabel("Neurons")
plt.ylabel(target)
plt.plot(data["NEURONS"], data[target], label = "Series", color = "black", linewidth = 1.0)
plt.plot(data_gn1["NEURONS"], data_gn1[target], label = "Series + GN 0.05%", linewidth = 1.0)
plt.fill_between(data["NEURONS"], data[target], data_gn1[target])
plt.plot(data_gn2["NEURONS"], data_gn2[target], label = "Series + GN 0.10%", linewidth = 1.0)
plt.fill_between(data["NEURONS"], data_gn1[target], data_gn2[target])
plt.plot(data_gn3["NEURONS"], data_gn3[target], label = "Series + GN 0.15%", linewidth = 1.0)
plt.fill_between(data["NEURONS"], data_gn2[target], data_gn3[target])
plt.plot(data_gn4["NEURONS"], data_gn4[target], label = "Series + GN 0.20%", linewidth = 1.0)
plt.fill_between(data["NEURONS"], data_gn3[target], data_gn4[target])
plt.legend()
plt.yscale('log')
plt.show()

#plt.figure(figsize = (10,10))
#plt.title("Henon Map")
#plt.xlabel("Quantidade de neurônios")
#plt.ylabel("MAPE")
#plt.plot(data["NEURONS"], data["MAPE"], label = "Series", color = "black")
#plt.plot(data_gn1["NEURONS"], data_gn1["MAPE"], label = "Series + GN 5%")
#plt.plot(data_gn2["NEURONS"], data_gn2["MAPE"], label = "Series + GN 10%")
#plt.plot(data_gn3["NEURONS"], data_gn3["MAPE"], label = "Series + GN 15%")
#plt.plot(data_gn4["NEURONS"], data_gn4["MAPE"], label = "Series + GN 20%")
#plt.legend()
#plt.show()