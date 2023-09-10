import pandas
import numpy
from sklearn.preprocessing import PowerTransformer

def split_df(df):
    spt = int(0.7 * df.shape[0])
    train = pandas.DataFrame(numpy.nan, index = df.index, columns = df.columns)
    test = train.copy(deep = True)
    for t in range(df.shape[0]):
        if t <= spt:
            train.iloc[t, :] = df.iloc[t, :]
        else:
            test.iloc[t, :] = df.iloc[t, :]
    return(train.dropna(axis = 0), test.dropna(axis = 0))

sets = "C:\\Users\\kgper\\OneDrive\\Área de Trabalho\\Neural Networks Codes\\ReservoirComputing\\DADOS REAIS\\"

df = pandas.read_csv(sets + "WEGE3.txt").astype("string")

df = df[(df.MINUTE.str.endswith("5")) | (df.MINUTE.str.endswith("0"))][["HIGH", "OPEN", "CLOSE", "LOW"]].astype(numpy.float64).round(2)
df = numpy.log10(df)

print(df)

log_ret = pandas.DataFrame(numpy.nan, index = df.index, columns = df.columns)

for t in range(1, df.shape[0]):
    log_ret.iloc[t,:] = df.iloc[t,:] - df.iloc[t - 1,:]

log_ret = log_ret.dropna(axis = 0)

print(log_ret)

log_ret.to_csv("WEGEData.txt", index = False, header = False)
#final["SP500"].to_csv("SP500Data.txt", index = False, header=False)
#final["HANGSENG"].to_csv("HangSengData.txt", index = False, header=False)
#final["DAX"].to_csv("DaxData.txt", index = False, header=False)
#final["BOVESPA"].to_csv("BovespaData.txt", index = False, header=False)
