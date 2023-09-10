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

btc = pandas.read_csv("DADOS REAIS\\DAILY_DATA_BITCOIN.csv", index_col = "DATE")["CLOSE"]
eth = pandas.read_csv("DADOS REAIS\\DAILY_DATA_ETHEREUM.csv", index_col = "DATE")["CLOSE"]
car = pandas.read_csv("DADOS REAIS\\DAILY_DATA_CARDANO.csv", index_col = "DATE")["CLOSE"]
dog = pandas.read_csv("DADOS REAIS\\DAILY_DATA_DOGECOIN.csv", index_col = "DATE")["CLOSE"]

print(btc)
print(car)

df = pandas.concat([btc, eth, car, dog], axis = 1).dropna(axis = 0)
df.columns = ["BITCOIN", "ETHEREUM", "CARDANO", "DOGECOIN"]
df = numpy.log10(df)

print(df)

log_ret = pandas.DataFrame(numpy.nan, index = df.index, columns = df.columns)

for t in range(1, df.shape[0]):
    log_ret.iloc[t,:] = df.iloc[t,:] - df.iloc[t - 1,:]

log_ret = log_ret.dropna(axis = 0)

print(log_ret)

final = log_ret.copy(deep = True)

pt = PowerTransformer()

train = split_df(final)[0]

train["BITCOIN"] = pt.fit_transform(train["BITCOIN"].values.reshape(-1, 1))
train["ETHEREUM"] = pt.fit_transform(train["ETHEREUM"].values.reshape(-1, 1))
train["CARDANO"] = pt.fit_transform(train["CARDANO"].values.reshape(-1, 1))
train["DOGECOIN"] = pt.fit_transform(train["DOGECOIN"].values.reshape(-1, 1))

final["BITCOIN"] = pt.fit_transform(final["BITCOIN"].values.reshape(-1, 1))
final["ETHEREUM"] = pt.fit_transform(final["ETHEREUM"].values.reshape(-1, 1))
final["CARDANO"] = pt.fit_transform(final["CARDANO"].values.reshape(-1, 1))
final["DOGECOIN"] = pt.fit_transform(final["DOGECOIN"].values.reshape(-1, 1))

test = split_df(final)[1]

print(train)
print(test)

train["BITCOIN"].to_csv("TrainData_Bitcoin.txt", header = None, index = False)
train["ETHEREUM"].to_csv("TrainData_Ethereum.txt", header = None, index = False)
train["CARDANO"].to_csv("TrainData_Cardano.txt", header = None, index = False)
train["DOGECOIN"].to_csv("TrainData_Dogecoin.txt", header = None, index = False)

test["BITCOIN"].to_csv("TestData_Bitcoin.txt", header = None, index = False)
test["ETHEREUM"].to_csv("TestData_Ethereum.txt", header = None, index = False)
test["CARDANO"].to_csv("TestData_Cardano.txt", header = None, index = False)
test["DOGECOIN"].to_csv("TestData_Dogecoin.txt", header = None, index = False)