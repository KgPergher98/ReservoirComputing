import numpy
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# GARANTE O FORMATO DAS COLUNAS PARA ARRAYS UNIDIMENSIONAIS
def reshape(df):
    # CASO NÃO HAJA INFORMAÇÕES EM SHAPE, EX.: SHAPE(X) = (#, )
    if len(df.shape) < 2:
        # FORÇA O FORMATO SHAPE(X) = (#, 1)
        return numpy.reshape(df, (df.shape[0], 1))
    else:
        # SENÃO, NÃO ALTERE O FORMATO DOS DADOS
        return df

# MATRIZ DE PESOS DE ENTRADA
def input_weights(N = 10, K = 1, input_scaling = 1):
    # CRIA UMA MATRIZ DE DIMENSÕES N x (K + 1)
    # DISTRIBUIÇÃO DE VALORES UNIFORMES ENTRE (SCALE x -0.5, SCALE x 0.5)
    return((numpy.random.rand(N, 1 + K) - 0.5) * input_scaling)

# MATRIZ DE PESOS DE RESERVATÓRIO
def reservoir_weights(N = 10, spectral_radius = 1.00, sparsity = 0):
    # CRIA UMA MATRIZ DE DIMENSÕES N x N
    # DISTRIBUIÇÃO DE VALORES UNIFORMES ENTRE (-0.5, 0.5)
    w = numpy.random.rand(N, N) - 0.5
    # CALCULA O MAIOR AUTOVALOR ABSOLUTO DA MATRIZ DE PESOS
    # FORÇA O AUTOVALOR DA MATRIZ PARA 1 E ESCALONA PELO RAIO ESPECTRAL DESEJADO
    w *= spectral_radius / numpy.max(numpy.abs(linalg.eig(w)[0]))
    # APLICA ESPARSIDADE NA MATRIZ
    w[numpy.random.rand(N, N) < sparsity] = 0
    return(w)

# MATRIZ DE ESTADOS
def create_states(dim = 1, patterns = 100):
    # CRIA UMA MATRIZ DE ESTADOS ZERADA
    return(numpy.zeros((dim, patterns)))

# HARVESTING -> ATUALIZAÇÃO DOS ESTADOS
def harvesting_states(state_matrix, input_patterns, win, wres, initial_state, leaking = 0.5, transiente = 0):
    # ESTADO INICIAL DO MODELO, EM CASO DE TRAINNING O ESTADO É POR DEFAULT ZERO PARA TODOS OS NEURÔNIOS
    state = initial_state.copy()
    for n in range(input_patterns.shape[0]):
        # CADA PADRÃO DE INPUT
        u = input_patterns[n,:]
        # ATUALIZA OS ESTADOS CONFORME A EQUAÇÃO DE ATUALIZAÇÃO
        state = (1 - leaking)*state + leaking*numpy.tanh(numpy.dot(win, reshape(numpy.hstack((1, u)))) + numpy.dot(wres, state))
        if n >= transiente:
            # DESCARTA O TRANSIENTE, REPASSA OS VALORES DE ESTADOS OBTIDOS
            state_matrix[:, n - transiente] = numpy.hstack((1, u, state[:,0]))
    return(state_matrix)

# REGRESSÃO DE RIDGE
def ridge_regression(states, target, reg = 1e-8):
    # RESOLVE A PROJEÇÃO DOS ESTADOS NO "TARGET" (OUTPUT DESEJADO)
    return(linalg.solve(numpy.dot(states, states.T) + reg * numpy.eye(states.shape[0]), 
                        numpy.dot(states, target.T)).T[0]
          )

# ALTERA O DATASET ORIGINAL, CONFORME O NÚMERO DE JANELAS DESEJADAS
def set_window(name = "NormHenon.txt", window_len = 1, begin = 0):
    # REALIZA A LEITURA DO ARQUIVO
    df = reshape(numpy.loadtxt(name, delimiter = ","))
    # CORTA O DATAFRAME NO PONTO DESEJADO
    df = df[begin:df.shape[0],:]
    # CALCULA TAMANHO DE JANELA x DIMENSÃO DA JANELA
    x = df.shape[1] * window_len
    # CRIA REFERÊNCIA INICIAL
    windows = df.copy()[window_len:df.shape[0],:]
    for j in range(1, window_len):
        # CONCATENA O DATAFRAME SHIFITADO
        windows = numpy.hstack([df.copy()[window_len - j:df.shape[0] - j,:], windows])
    return(windows, x)

def set_window2(name = "NormHenon.txt", window_len = 1, begin = 0, steps = 0):
    # REALIZA A LEITURA DO ARQUIVO
    df = reshape(numpy.loadtxt(name, delimiter = ","))
    # CORTA O DATAFRAME NO PONTO DESEJADO
    df = df[begin:df.shape[0],:]
    # CALCULA TAMANHO DE JANELA x DIMENSÃO DA JANELA
    x = df.shape[1] * window_len
    # CRIA REFERÊNCIA INICIAL
    windows = df.copy()[window_len - 1 : df.shape[0] - steps,:]
    for j in range(1, window_len):
        # CONCATENA O DATAFRAME SHIFITADO
        windows = numpy.hstack([df.copy()[window_len - j - 1: df.shape[0] - j - steps,:], windows])
    return(windows, x)

# AVALIA O MODELO
def model_metrics(y, y_target):
    mts = [numpy.nan] * 12
    mse = mean_squared_error(y, y_target)
    rmse = mean_squared_error(y, y_target, squared = False)
    mape = mean_absolute_percentage_error(y, y_target)
    mts[0] = mse
    mts[1] = rmse
    mts[2] = mape
    if y.shape[0] < 2:
        mts[3] = mse
        mts[4] = rmse
        mts[5] = mape
    else:
        mts[3] = mean_squared_error(y[0,:], y_target[0,:])
        mts[4] = mean_squared_error(y[0,:], y_target[0,:], squared = False)
        mts[5] = mean_absolute_percentage_error(y[0,:], y_target[0,:])
    if y.shape[0] >= 2:
        mts[6] = mean_squared_error(y[1,:], y_target[1,:])
        mts[7] = mean_squared_error(y[1,:], y_target[1,:], squared = False)
        mts[8] = mean_absolute_percentage_error(y[1,:], y_target[1,:])
    if y.shape[0] >= 3:
        mts[9] = mean_squared_error(y[2,:], y_target[2,:])
        mts[10] = mean_squared_error(y[2,:], y_target[2,:], squared = False)
        mts[11] = mean_absolute_percentage_error(y[2,:], y_target[2,:])
    return(mts)