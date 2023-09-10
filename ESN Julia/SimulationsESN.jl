#=
    TESTAGEM DO USO DE ECHO STATE NEURAL NETWORKS COM LEAKING RATE 
    TESTAGEM COM O USO DE SISTEMAS CAÓTICOS PADRÃO
    MACKEY GLASS, MAPA DE HÉNON E ATRATOR DE LORENZ 

    CRIADO EM 03/07/2022
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
#Random.seed!(17)
# Treino Noise
#Random.seed!(667)
# Modelo I
#Random.seed!(42)
# Modelo II
#Random.seed!(134)
# Modelo III
Random.seed!(349)

# VALIDACAO
#==#
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


# TREINO
#=
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
=#

io = open("ParametrosMackey22TesteModeloIII.txt", "a")
writedlm(io, ["TRAIN" "NEURONS" "RADIUS" "LEAKING" "SPARSITY" "SCALE" "MSE" "NMSE" "MAPE"], ',')
#writedlm(io, ["TRAIN" "NEURONS" "RADIUS" "LEAKING" "SPARSITY" "SCALE" "MSE" "NMSE" "MAPE" "MSE(X)" "NMSE(X)" "MAPE(X)" "MSE(Y)" "NMSE(Y)" "MAPE(Y)"], ',')
#writedlm(io, ["TRAIN" "NEURONS" "RADIUS" "LEAKING" "SPARSITY" "SCALE" "MSE" "NMSE" "MAPE" "MSE(X)" "NMSE(X)" "MAPE(X)" "MSE(Y)" "NMSE(Y)" "MAPE(Y)" "MSE(Z)" "NMSE(Z)" "MAPE(Z)"], ',')

# ITERACOES COM DISTINTOS RUIDOS
noise_iter = 10

for iter in 1:noise_iter

    println("Iteracao: ", iter)

    # DEFINE O SISTEMA A SER ANALISADO
    dinsystem = readdlm("NormMackey22.txt", ',')

    # INTENSIDADE DO RUIDO
    #noise_factor = 0.00001
    #noise_factor = 0.05
    #noise_factor = 0.001
    noise_factor = 0.001

    if noise_factor > 0
        # GERACAO DE SERIE DE RUIDO
        noise = rand(Normal(0.0, 1.0), size(dinsystem)) .* noise_factor
        # DISTORCE A SERIE ORIGINAL
        dinsystem = dinsystem .+ noise
    end

    # DEFINE O QUE É INPUT E O QUE É OUTPUT - ATRAVÉS DAS DIMENSÕES DO SISTEMA
    u = dinsystem
    y = dinsystem
    # TOTAL DE DADOS PRESENTES PARA A ANÁLISE (TAU)
    total_data = size(dinsystem)[1]
    dinsystem_dim = size(dinsystem)[2]
    #noise_dim = size(noise)[2]

    # DIMENSÕES DO SISTEMA
    input_dim = size(u)[2]
    output_dim = size(y)[2]

    # OBSERVAÇÕES PARA TESTE
    testing_length = 1::Int64

    # QUANTIDADE DE JANELAS
    window = 200::Int64
    #window = total_data - 11

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
                                    t0 = 10 + 200
                                    #t0 = 10
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
                                # CALCULA O ERRO FINAL PARA X
                                #mse_x = mean_squared_error(data = test_matrix[:,1], pred_data = pred_matrix[:,1])
                                #nmse_x = normalized_mean_squared_error(data = test_matrix[:,1], pred_data = pred_matrix[:,1])
                                #mape_x = mean_absolute_percentage_error(data = test_matrix[:,1], pred_data = pred_matrix[:,1])
                                # CALCULA O ERRO FINAL PARA Y
                                #mse_y = mean_squared_error(data = test_matrix[:,2], pred_data = pred_matrix[:,2])
                                #nmse_y = normalized_mean_squared_error(data = test_matrix[:,2], pred_data = pred_matrix[:,2])
                                #mape_y = mean_absolute_percentage_error(data = test_matrix[:,2], pred_data = pred_matrix[:,2])
                                # CALCULA O ERRO FINAL PARA Y
                                #mse_z = mean_squared_error(data = test_matrix[:,3], pred_data = pred_matrix[:,3])
                                #nmse_z = normalized_mean_squared_error(data = test_matrix[:,3], pred_data = pred_matrix[:,3])
                                #mape_z = mean_absolute_percentage_error(data = test_matrix[:,3], pred_data = pred_matrix[:,3])
                                # ARQUIVO
                                writedlm(io, [training_length n_neurons spectral_radius leaking_rate sparsity input_scale mse nmse mape], ',')
                                #writedlm(io, [training_length n_neurons spectral_radius leaking_rate sparsity input_scale mse nmse mape mse_x nmse_x mape_x mse_y nmse_y mape_y], ',')
                                #writedlm(io, [training_length n_neurons spectral_radius leaking_rate sparsity input_scale mse nmse mape mse_x nmse_x mape_x mse_y nmse_y mape_y mse_z nmse_z mape_z], ',')
                            end

                        end

                    end

                end

            end

        end

    end

end

close(io)





#=
noise = readdlm("gaussianNoise3D.txt", ',')



# VARIÂNCIAS
#var_x = var(dinsystem[:,1])
#var_y = var(dinsystem[:,2])
#var_z = var(dinsystem[:,3])

#var_noise_x = var(noise[:,1])
#var_noise_y = var(noise[:,2])
#var_noise_z = var(noise[:,3])

# NORMALIZAÇÃO DA VARIÂNCIA E REATRIBUIÇÃO
#noise_factor = 0.00001
noise_factor = 0.0001
noise[:,1] = (noise[:,1]) .* (noise_factor)
noise[:,2] = (noise[:,2]) .* (noise_factor)
noise[:,3] = (noise[:,3]) .* (noise_factor)

#noise[:,1] = (noise[:,1] ./ var_noise_x) .* (noise_factor * var_x)
#noise[:,2] = (noise[:,2] ./ var_noise_y) .* (noise_factor * var_y)
#noise[:,3] = (noise[:,3] ./ var_noise_z) .* (noise_factor * var_z)
=#