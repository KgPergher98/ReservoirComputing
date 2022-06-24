"""
    Classe Echo State Neural Network
    Criada em 15 de abril de 2022
    Kevin Pergher
"""

from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import numpy
import scipy
import SistemasCaoticos


class EchoStateNeuralNetwork():

    def reshaping_data(data):
        if data.ndim < 2:
            data = numpy.reshape(data, (len(data), 1))
        return data

    def __init__(self, function, number_reservoir=100, spectral_radius=0.9,
                 training=None, testing=2000, sparsity=0, noise=0.001, seed=42, leaking_rate=0.3, message=False):

        # Dados a serem utilizados para a criação da rede
        # Seed controla a "geração de um estado random", possibilitando reproduzir o experimento
        self.seed = seed
        self.random_state = numpy.random.RandomState(self.seed)

        # Controle das mensagens para visualização pelo usuário do que ocorre, útil para debug
        self.message = message

        self.function = function
        self.function_extention = self.function.shape[0]
        if function.ndim >= 2:
            self.function_dimension = self.function.shape[1]
        else:
            self.function_dimension = 1
        self.init_length = 100

        # Tamanho da série a ser utilizada no treino e no teste
        if type(training) == int:
            self.training_data_len = training
        else:
            self.training_data_len = self.function_extention
        self.test_data_len = testing

        # Dimensões dos dados da entrada
        self.input_dimension = self.function_dimension
        # Dimensão dos dados de saída
        self.output_dimension = self.function_dimension
        # Número de "neurônios" no reservatório
        self.number_reservoir = number_reservoir
        # Raio espectral a ser utilizado
        self.spectral_radius = spectral_radius
        # Esparcidade da matriz de pesos internos
        self.sparsity = sparsity
        # Ruído a ser acrescido
        self.noise = noise
        # Vazamento a ser utilizado
        self.leaking_rate = leaking_rate

        if self.message:
            print("\nInicializando ESN\n > Número de Reservatórios (N): ",self.number_reservoir)
            print(" > Dimensão de entrada: ",self.input_dimension, ", K = ", self.training_data_len, "\n > Dimensão de saída: ", self.output_dimension, ", L = ", self.training_data_len)
            print(" > Dimensão da função de entada: ", function.shape, "\n > Dados para treinamento: ", self.training_data_len)


    def create_reservoir(self):
        # Cria a matriz aleatória de pesos de reservoir com n_res x n_res e pesos em [-0.5, 0.5]
        self.w_reservoir = self.random_state.rand(self.number_reservoir, self.number_reservoir) - 0.5
        # Aplica 0s em um número determinado de conexões entre neurônios 
        self.w_reservoir[self.random_state.rand(*self.w_reservoir.shape) < self.sparsity] = 0
        # Procura o maior eigenvalue em valor absoluto
        max_eigenvalue = numpy.max(numpy.abs(numpy.linalg.eigvals(self.w_reservoir)))
        # Realiza o scale em relação ao raio spectral e ao máximo eigenvalue
        self.w_reservoir = self.w_reservoir * (self.spectral_radius/ max_eigenvalue)

        # Pesos de input, dimensão n_res x inp_dim, valores entre [-1, 1]
        self.w_input = (self.random_state.rand(self.number_reservoir, self.input_dimension+1) * 2) - 1

        # Pesos de feedback, dimensão n_res x out_dim, valores entre [-1, 1]
        self.w_feedback = (self.random_state.rand(self.number_reservoir, self.output_dimension) * 2) - 1

        self.teacher_output = self.function[None, self.init_length+1:self.training_data_len+1]


        # Cria a matriz de estados
        self.states = numpy.zeros((self.number_reservoir+self.input_dimension+1,self.training_data_len-self.init_length))

        if self.message:
            print("Criação do reservatório\n > Dimensões do reservatório: ", self.w_reservoir.shape, "\n > Dimensão dos pesos de entrada: ", self.w_input.shape)
            print(" > Dimensão dos pesos de feedback: ", self.w_feedback.shape)
            print(" > Dimensão do teacher de saída: ", self.teacher_output.shape, "\n > Dimensão da matriz de estados: ", self.states.shape)


    def update_states_with_feedback(self, round):
        # Calcula a preativação: X(n+1) = Wres . X(n) + Win . u(n+1) + Wfeed . y(n)
        preactivation = numpy.dot(self.w_reservoir, self.states[round-1,:])
        preactivation += numpy.dot(self.w_input, self.teacher_input[round,:])
        #preactivation += numpy.dot(self.w_feedback, self.teacher_output[round-1,:]) 
        # Incide um ruído branco sobre os estado X(n+1)
        preactivation = preactivation + (self.noise * (self.random_state.rand(self.number_reservoir) - 0.5))
        # Retorna a "ativação" dos estados
        return numpy.tanh(preactivation)

    def update_states(self, input, last_state):
        # Calcula a preativação: 
        preactivation = numpy.dot(self.w_reservoir, last_state)
        preactivation += numpy.dot(self.w_input, input)
        # Incide um ruído branco sobre os estado X(n+1)
        # Retorna a "ativação" dos estados
        return numpy.tanh(preactivation)

    def train_model(self):
        # Atualização dos estados no perído de trainamento
        self.last_state = numpy.zeros((self.number_reservoir,1))
        for round in range(self.training_data_len):
            u = self.function[round]
            self.last_state = (1-self.leaking_rate)*self.last_state + self.leaking_rate*EchoStateNeuralNetwork.update_states(self, input=numpy.vstack((1,u)), last_state=self.last_state)
            if round >= self.init_length:
                # Realiza a atualização "linha a linha", que equivale aos pasos discretos no tempo
                self.states[:,round-self.init_length] = numpy.vstack((1,u,self.last_state))[:,0]
        
        reg = 1e-8
        self.w_out = scipy.linalg.solve(numpy.dot(self.states, self.states.T) + reg*numpy.eye(1+self.input_dimension+self.number_reservoir),
                                        numpy.dot(self.states, self.teacher_output.T)).T
        if self.message:
            print("Treinamento do modelo\n > Atualização dos estados de {} -> {}".format(self.init_length, self.training_data_len))
            print(" > Dimensão do pesos de saída: ", self.w_out.shape)

    def model_prediction(self):
        self.prediction = numpy.zeros((self.output_dimension, self.test_data_len))
        u = self.function[self.training_data_len]

        for t in range(self.test_data_len):
            self.last_state = (1-self.leaking_rate)*self.last_state + self.leaking_rate*EchoStateNeuralNetwork.update_states(self, input=numpy.vstack((1,u)), last_state=self.last_state)
            self.last_state 
            prediction = numpy.dot(self.w_out, numpy.vstack((1,u,self.last_state)))
            self.prediction[:,t] = prediction
            u = prediction


    def metrics(self, y=0, yt=0, outside=False):
        if outside:
            print("Métricas obtidas:\n > MSE Teste: ",mean_squared_error(y,yt))
        else:
            print("Métricas obtidas:\n > MSE Teste: ",mean_squared_error(self.function[self.training_data_len+1:self.training_data_len+1+self.test_data_len], 
                                                                     self.prediction[0,:]))

if __name__ == "__main__":

    def metrics(y=0, yt=0):
        print("Métricas obtidas:\n > MSE Teste: ",mean_squared_error(y,yt))

    def single_axis_plot(system, prediction, ref, title, plot=False):
        fig, ax = pyplot.subplots(2, figsize=(15,10))
        ax[0].plot(ref,system, label="Valor Esperado", color="blue")
        ax[0].legend()
        ax[1].plot(ref,prediction, label="Previsão", color="red")
        ax[1].legend()
        if plot:
            pyplot.show()
        pyplot.savefig(title)
        pyplot.close()

    def map_plot(system_x, system_y, prediction_x, prediction_y, title, plot=False):
        fig, ax = pyplot.subplots(1, 2, figsize=(15,10))
        ax[0].scatter(system_x, system_y, label="Valor Esperado", color="blue")
        ax[0].legend()
        ax[1].scatter(prediction_x, prediction_y, label="Previsão", color="red")
        ax[1].legend()
        if plot:
            pyplot.show()
        pyplot.savefig(title)
        pyplot.close()

    def spatial_plot(system_x, system_y, system_z, prediction_x, prediction_y, prediction_z, title, plot=False):
        fig = pyplot.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(system_x, system_y, system_z, label="Valor Esperado", color="blue")
        ax.legend()
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(prediction_x, prediction_y, prediction_z, label="Previsão", color="red")
        ax.legend()
        if plot:
            pyplot.show()
        pyplot.savefig(title)
        pyplot.close()

    def train_network(x, test, number_res, train, spec_radius=1.25, message=False):
        nnx = EchoStateNeuralNetwork(function = x, 
                                    testing = test, 
                                    number_reservoir = number_res,
                                    training = train,
                                    spectral_radius = spec_radius,
                                    message = False)
        nnx.create_reservoir()
        nnx.train_model()
        nnx.model_prediction()
        #nnx.metrics()
        return nnx.prediction.T

    def single_nn_test(data="Henon_Map", observations=10000, train=6000, test=10, number_res=1000, plot=False, add_info=""):
        print("\nIniciando free running: ", data)
        if data == "Henon_Map":
            function = SistemasCaoticos.henon_map(x0=0, y0=0, n_observations=observations)
            x = function[:,0]
            y = function[:,1]
        elif data == "Lorenz_Attractor":
            function = SistemasCaoticos.lorenz_attractor(n_observations=observations, final_t=40)
            x = function[:,0]
            y = function[:,1]
            z = function[:,2]
        elif data == "Mackey_Glass":
            function = SistemasCaoticos.mackey_glass(n_observations=observations, tau=17, gama=0.9)
            x = function[:,0]
        prediction_x = train_network(x, test=test, number_res=number_res, train=train, spec_radius=1.25, message=False)
        if data in ["Henon_Map", "Lorenz_Attractor"]:
            prediction_y = train_network(y, test=test, number_res=number_res, train=train, spec_radius=1.25, message=False)
        if data in ["Lorenz_Attractor"]:
            prediction_z = train_network(z, test=test, number_res=number_res, train=train, spec_radius=1.25, message=False)


        if data in ["Mackey_Glass"]:
            metrics(prediction_x, x[train+1:train+test+1])
            single_axis_plot(system=x[train+1:train+test+1], prediction=prediction_x, ref=numpy.arange(0,test), title= data + add_info + "_X.png", plot=plot)
            pyplot.figure()
            pyplot.plot(x)
            pyplot.show()
        elif data in ["Henon_Map"]:
            metrics(prediction_x, x[train+1:train+test+1])
            metrics(prediction_y, y[train+1:train+test+1])
            metrics(numpy.column_stack((prediction_x,prediction_y)), numpy.column_stack((x[train+1:train+test+1],y[train+1:train+test+1])))
            single_axis_plot(system=x[train+1:train+test+1], prediction=prediction_x, ref=numpy.arange(0,test), title= data + add_info + "_X.png", plot=plot)
            single_axis_plot(system=y[train+1:train+test+1], prediction=prediction_y, ref=numpy.arange(0,test), title= data + add_info + "_Y.png", plot=plot)
            map_plot(x[train+1:train+test+1], y[train+1:train+test+1], prediction_x, prediction_y, title= data + add_info + ".png", plot=plot)
        elif data in ["Lorenz_Attractor"]:
            metrics(prediction_x, x[train+1:train+test+1])
            metrics(prediction_y, y[train+1:train+test+1])
            metrics(prediction_z, z[train+1:train+test+1])
            metrics(numpy.column_stack((prediction_x,prediction_y, prediction_z)), 
                        numpy.column_stack((x[train+1:train+test+1],y[train+1:train+test+1],z[train+1:train+test+1])))
            single_axis_plot(system=x[train+1:train+test+1], prediction=prediction_x, ref=numpy.arange(0,test), title= data + add_info + "_X.png", plot=plot)
            single_axis_plot(system=y[train+1:train+test+1], prediction=prediction_y, ref=numpy.arange(0,test), title= data + add_info + "_Y.png", plot=plot)
            map_plot(x[train+1:train+test+1], y[train+1:train+test+1], prediction_x, prediction_y, title= data + add_info + ".png", plot=plot)
            spatial_plot(x[train+1:train+test+1], y[train+1:train+test+1], z[train+1:train+test+1], 
                         prediction_x, prediction_y, prediction_z, title= data + "_3D.png", plot=plot)

    def moving_nn_test(data="Henon_Map", observations=10000, train=1000, test=1, n_data=1000, number_res=200, plot=False, add_info=""):
        print("\nIniciando janelas móveis: ", data)

        # Função original - Base de dados
        if data=="Henon_Map":
            function = SistemasCaoticos.henon_map(x0=0, y0=0, n_observations=observations)
            x = function[:,0]
            y = function[:,1]
        elif data == "Mackey_Glass":
            function = SistemasCaoticos.mackey_glass(n_observations=observations, tau=22)
            x = function[:,0]
        elif data == "Lorenz_Attractor":
            function = SistemasCaoticos.lorenz_attractor(n_observations=observations, final_t=40)
            x = function[:,0]
            y = function[:,1]
            z = function[:,2]

        fx1 = numpy.zeros((n_data,))
        fy1 = numpy.zeros((n_data,))
        fz1 = numpy.zeros((n_data,))
        for t in range(n_data):
            time = observations - n_data + t - 1
            fx1[t] = train_network(x[time-train:time+1], test=test, number_res=number_res, train=train, spec_radius=1.25, message=False)[0,0]
            if data in ["Henon_Map", "Lorenz_Attractor"]:
                fy1[t] = train_network(y[time-train:time+1], test=test, number_res=number_res, train=train, spec_radius=1.25, message=False)[0,0]
            if data in ["Lorenz_Attractor"]:
                fz1[t] = train_network(z[time-train:time+1], test=test, number_res=number_res, train=train, spec_radius=1.25, message=False)[0,0]       
        if data in ["Mackey_Glass"]:
            metrics(fx1, x[observations-n_data:observations])
            single_axis_plot(system=x[observations-n_data:observations], prediction=fx1, ref=numpy.arange(0,n_data), title="MM_" + data + add_info + "_X.png", plot=plot)
        elif data in ["Henon_Map"]:
            metrics(fx1, x[observations-n_data:observations])
            metrics(fy1, y[observations-n_data:observations])
            metrics(numpy.column_stack((fx1,fy1)), numpy.column_stack((x[observations-n_data:observations],y[observations-n_data:observations])))
            single_axis_plot(system=x[observations-n_data:observations], prediction=fx1, ref=numpy.arange(0,n_data), title="MM_" + data + add_info + "_X.png", plot=plot)
            single_axis_plot(system=y[observations-n_data:observations], prediction=fy1, ref=numpy.arange(0,n_data), title="MM_" + data + add_info + "_Y.png", plot=plot)
            map_plot(x[observations-n_data:observations], y[observations-n_data:observations], fx1, fy1, title="MM_" + data + add_info + ".png", plot=plot)
        elif data in ["Lorenz_Attractor"]:
            metrics(fx1, x[observations-n_data:observations])
            metrics(fy1, y[observations-n_data:observations])
            metrics(fz1, z[observations-n_data:observations])
            metrics(numpy.column_stack((fx1,fy1, fz1)), 
                        numpy.column_stack((x[observations-n_data:observations],y[observations-n_data:observations],z[observations-n_data:observations])))
            single_axis_plot(system=x[observations-n_data:observations], prediction=fx1, ref=numpy.arange(0,n_data), title="MM_" + data + add_info + "_X.png", plot=plot)
            single_axis_plot(system=y[observations-n_data:observations], prediction=fy1, ref=numpy.arange(0,n_data), title="MM_" + data + add_info + "_Y.png", plot=plot)
            map_plot(x[observations-n_data:observations], y[observations-n_data:observations], fx1, fy1, title="MM_" + data + add_info + ".png", plot=plot)
            spatial_plot(x[observations-n_data:observations], y[observations-n_data:observations], z[observations-n_data:observations], 
                         fx1, fy1, fz1, title= data + "_3D.png", plot=plot)

    single_nn_test(data="Henon_Map", observations=10000, train=1000, test=400, number_res=800, add_info="")
    moving_nn_test(data="Henon_Map", observations=10000, train=200, test=1, n_data=400, number_res=800, add_info="")
    single_nn_test(data="Mackey_Glass", observations=10000, train=2000, test=400, number_res=1000, add_info="_t17")
    moving_nn_test(data="Mackey_Glass", observations=10000, train=300, test=1, n_data=400, number_res=1000, add_info="_t22")
    single_nn_test(data="Lorenz_Attractor", observations=10000, train=1000, test=800, number_res=800, add_info="")
    moving_nn_test(data="Lorenz_Attractor", observations=10000, train=200, test=1, n_data=800, number_res=800, add_info="")