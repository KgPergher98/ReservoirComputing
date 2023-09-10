#=
    TESTAGEM DO USO DE ECHO STATE NEURAL NETWORKS COM LEAKING RATE 
    DADOS REAIS

    CRIADO EM 16/01/2023
    KEVIN PERGHER
=#

# PRINCIPAIS BIBLIOTECAS A SEREM UTILIZADAS NA EXECUÇÃO DESTE PROGRAMA
using Random
using Distributions
using DelimitedFiles
using LinearAlgebra
using Statistics
using Plots
using CSV

# ARQUIVOS DE FUNÇÕES E OUTROS, PRESENTES NO DIRETÓRIO DO PROGRAMA
include(pwd() * "\\LeakyRidgeESN.jl")
include(pwd() * "\\MeasuresESN.jl")

println("\nESN > Echo State Neural Network")

# DEFINE A SEED DO ALGORITMO PSEUDO-ALEATÓRIO
# Train
Random.seed!(98)
# Test
#Random.seed!(713)


# VALIDACAO
#=
trainning_set = [10, 20, 30, 40, 50, 
                 60, 70, 80, 90, 100,
                 110, 120, 130, 140, 150, 
                 160, 170, 180, 190, 200]
neurons_set = [20, 40, 60, 80, 100,
               120, 140, 160, 180, 200,
               220, 240, 260, 280, 300,
               320, 340, 360, 380, 400]

sparsity_set = [0.9]
radius_set = [0.6]
leaking_set = [0.3]
scaling_set = [0.4]
=#

# TREINO
#==#
trainning_set = [10]
neurons_set = [20]

radius_set = [0.2, 0.4, 0.6, 0.8, 1.0,
              1.2, 1.4, 1.6, 1.8, 2.0,
              2.2, 2.4, 2.6, 2.8, 3.0]
leaking_set = [0.1, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9]
sparsity_set = [0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9]
scaling_set = [0.2, 0.4, 0.6, 0.8, 1.0,
               1.2, 1.4, 1.6, 1.8, 2.0,
               2.2, 2.4, 2.6, 2.8, 3.0]


io = open("ParametrosRealTreinoDogecoin.txt", "a")
writedlm(io, ["TRAIN" "NEURONS" "RADIUS" "LEAKING" "SPARSITY" "SCALE" "MSE" "NMSE" "MAPE"], ',')

# DEFINE O SISTEMA A SER ANALISADO
dinsystem = readdlm("TrainData_Dogecoin.txt", ',')

# DEFINE O QUE É INPUT E O QUE É OUTPUT - ATRAVÉS DAS DIMENSÕES DO SISTEMA
u = dinsystem
y = dinsystem
# TOTAL DE DADOS PRESENTES PARA A ANÁLISE (TAU)
total_data = size(dinsystem)[1]
dinsystem_dim = size(dinsystem)[2]

# DIMENSÕES DO SISTEMA
input_dim = size(u)[2]
output_dim = size(y)[2]

# OBSERVAÇÕES PARA TESTE
testing_length = 1::Int64

# QUANTIDADE DE JANELAS
#window = 200::Int64
window = total_data - 11

# MENSAGENS
messages = true::Bool

@time begin

    # OBSERVAÇÕES PARA TREINAMENTO
    for training_length in trainning_set

        # TRANSIENTE
        transient = Int64(training_length/10)

        # NÚMERO DE UNIDADES DE PROCESSAMENTO (NEURÔNIOS)
        for n_neurons in neurons_set

            # RAIO ESPECTRAL
            for spectral_radius in radius_set

                # TAXA DE VAZAMENTO
                for leaking_rate in leaking_set

                    # ESPARCIDADE
                    for sparsity in sparsity_set

                        for input_scale in scaling_set

                            # DEFINIMOS UMA MATRIZ DE PESOS DE ENTRADA
                            w_input = input_weights(input_dim = input_dim, 
                                                    n_neurons = n_neurons,
                                                    input_scale = input_scale,
                                                    messages = false)

                            # DEFINIMOS UMA MATRIZ DE PESOS DE RESERVATÓRIO (CONEXÕES ENTRE NEURÔNIOS) 
                            w_reservoir = reservoir_weights(n_neurons = n_neurons, 
                                                            spectral_radius = spectral_radius,
                                                            sparsity = sparsity,
                                                            messages = false)

                            # MATRIZES DE COMPARAÇÃO (PREDITO VS REAL)
                            test_matrix = zeros(window, dinsystem_dim)
                            pred_matrix = zeros(window, dinsystem_dim)

                            # PARA CADA JANELA
                            for t in 1:window

                                # INICIAL ALEATÓRIO
                                t0 = 0
                                # ENTRADA PARA CADA RODADA DE TREINAMENTO
                                u_train = u[t0 + t : t0 + t + training_length - 1, :]

                                # OUTPUT DO TRAINAMENTO
                                y_train = y[t0 + t + transient + 1 : t0 + t + training_length, :]

                                # CALCULA OS ESTADOS EXTENDIDOS, SE GUARDA O ÚLTIMO ESTADO
                                local extended_states, last_state = harvest_states(w_input = w_input, 
                                                                                w_reservoir = w_reservoir, 
                                                                                u = u_train, 
                                                                                n_neurons = n_neurons, 
                                                                                input_dim = input_dim,
                                                                                training_length = training_length,
                                                                                transient = transient,
                                                                                leaking_rate = leaking_rate,
                                                                                messages = false)
                                # OUTPUT ATRAVÉS DA REGRESSÃO DE TIKHONOV
                                local w_output = tikhonov_regression(extended_states = extended_states, 
                                                                    output = y_train, 
                                                                    messages = false)

                                # PREDICTION
                                local y_hat = predict(w_output = w_output, 
                                                    w_input = w_input, 
                                                    w_reservoir = w_reservoir,
                                                    u = u[t0 + t + training_length, :], 
                                                    states = last_state, 
                                                    output_dim = output_dim, 
                                                    to_predict = testing_length, 
                                                    leaking_rate = leaking_rate, 
                                                    messages = false)

                                # ATRIBUI OS VALORES ENCONTRADOS
                                test_matrix[t, :] .= u[t0 + t + training_length + testing_length, :]
                                pred_matrix[t, :] .= y_hat[testing_length, :]

                            end 

                            # CALCULA O ERRO FINAL
                            mse = mean_squared_error(data = test_matrix, pred_data = pred_matrix)
                            nmse = normalized_mean_squared_error(data = test_matrix, pred_data = pred_matrix)
                            mape = mean_absolute_percentage_error(data = test_matrix, pred_data = pred_matrix)
                            
                            # ARQUIVO
                            writedlm(io, [training_length n_neurons spectral_radius leaking_rate sparsity input_scale mse nmse mape], ',')
                        end

                    end

                end

            end

        end

    end

end

close(io)