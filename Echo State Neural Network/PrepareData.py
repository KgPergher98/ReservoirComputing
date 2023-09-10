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
#btc = pandas.read_csv(sets + "SP500.txt", index_col = "Date")["CLOSE"]
#eth = pandas.read_csv(sets + "HANGSENG.txt", index_col = "Date")["CLOSE"]
#car = pandas.read_csv(sets + "DAX.txt", index_col = "Date")["CLOSE"]
#dog = pandas.read_csv(sets + "BOVESPA.txt", index_col = "Date")["CLOSE"]
btc = pandas.read_csv(sets + "BITCOIN.txt", index_col = "Date")["CLOSE"]
eth = pandas.read_csv(sets + "ETHEREUM.txt", index_col = "Date")["CLOSE"]
car = pandas.read_csv(sets + "CARDANO.txt", index_col = "Date")["CLOSE"]
dog = pandas.read_csv(sets + "DOGECOIN.txt", index_col = "Date")["CLOSE"]

#print(btc)
#print(car)

df = pandas.concat([btc, eth, car, dog], axis = 1).dropna(axis = 0)
#df.columns = ["SP500", "HANGSENG", "DAX", "BOVESPA"]
df.columns = ["BITCOIN", "ETHEREUM", "CARDANO", "DOGECOIN"]

#df["BITCOIN"].to_csv("BitcoinData.txt")
#df["ETHEREUM"].to_csv("EthereumData.txt")
#df["CARDANO"].to_csv("CardanoData.txt")
#df["DOGECOIN"].to_csv("DogecoinData.txt")

df = numpy.log10(df)

print(df)

log_ret = pandas.DataFrame(numpy.nan, index = df.index, columns = df.columns)

for t in range(1, df.shape[0]):
    log_ret.iloc[t,:] = df.iloc[t,:] - df.iloc[t - 1,:]

log_ret = log_ret.dropna(axis = 0)

print(log_ret)

final = log_ret.copy(deep = True)

pt = PowerTransformer()

#train = split_df(final)[0]

#train["BITCOIN"] = pt.fit_transform(train["BITCOIN"].values.reshape(-1, 1))
#train["ETHEREUM"] = pt.fit_transform(train["ETHEREUM"].values.reshape(-1, 1))
#train["CARDANO"] = pt.fit_transform(train["CARDANO"].values.reshape(-1, 1))
#train["DOGECOIN"] = pt.fit_transform(train["DOGECOIN"].values.reshape(-1, 1))

#final["BITCOIN"] = pt.fit_transform(final["BITCOIN"].values.reshape(-1, 1))
#final["ETHEREUM"] = pt.fit_transform(final["ETHEREUM"].values.reshape(-1, 1))
#final["CARDANO"] = pt.fit_transform(final["CARDANO"].values.reshape(-1, 1))
#final["DOGECOIN"] = pt.fit_transform(final["DOGECOIN"].values.reshape(-1, 1))

#final["BITCOIN"].to_csv("BitcoinData.txt", index = False, header=False)
#final["ETHEREUM"].to_csv("EthereumData.txt", index = False, header=False)
#final["CARDANO"].to_csv("CardanoData.txt", index = False, header=False)
#final["DOGECOIN"].to_csv("DogecoinData.txt", index = False, header=False)

print(final)
#final["SP500"].to_csv("SP500Data.txt", index = False, header=False)
#final["HANGSENG"].to_csv("HangSengData.txt", index = False, header=False)
#final["DAX"].to_csv("DaxData.txt", index = False, header=False)
#final["BOVESPA"].to_csv("BovespaData.txt", index = False, header=False)