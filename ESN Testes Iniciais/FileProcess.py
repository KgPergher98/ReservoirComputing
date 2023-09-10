import pandas
import numpy
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pandas.read_csv("lorenzAttractor.txt", header = None)

print(df)
print(df.shape)
print(type(df), end = "\n\n")

pw = preprocessing.PowerTransformer(method = "yeo-johnson")

"""
cdfx = df[0].to_numpy()
cdfx = numpy.reshape(cdfx, (cdfx.shape[0], 1))
print(cdfx)
print(cdfx.shape)
print(type(cdfx), end = "\n\n")

cdfy = df[1].to_numpy()
cdfy = numpy.reshape(cdfy, (cdfy.shape[0], 1))
print(cdfy)
print(cdfy.shape)
print(type(cdfy), end = "\n\n")

cdfz = df[2].to_numpy()
cdfz = numpy.reshape(cdfz, (cdfz.shape[0], 1))
print(cdfz)
print(cdfz.shape)
print(type(cdfz), end = "\n\n")

ndf_x = pw.fit_transform(cdfx)
ndf_y = pw.fit_transform(cdfy)
ndf_z = pw.fit_transform(cdfz)

print(ndf_x)
print(ndf_y)
"""

numpy.savetxt("NormLorenz.txt", pw.fit_transform(df), delimiter = ",")

#print(ndf)
#print(ndf.shape)
#print(type(ndf), end = "\n\n")

#plt.figure(figsize=(15,15))
#plt.plot(ndf.iloc[0:1000,0])
#plt.show()

#plt.figure(figsize=(15,15))
#plt.plot(ndf[0:1000,0])
#plt.show()

#fig = plt.figure(figsize=(15,15))
#ax = fig.add_subplot(projection='3d')
#ax.scatter(ndf_x[0:3000,0], ndf_y[0:3000,0], ndf_z[0:3000,0])
#plt.show()


#numpy.savetxt("NormLorenzX.txt", ndf_x)
#numpy.savetxt("NormLorenzY.txt", ndf_y)
#numpy.savetxt("NormLorenzZ.txt", ndf_z)