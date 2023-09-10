#=
    TESTES AIR FOURIER

    CRIADO EM 16/07/2022
    KEVIN PERGHER
=#

# PRINCIPAIS BIBLIOTECAS A SEREM UTILIZADAS NA EXECUÇÃO DESTE PROGRAMA
using Random
using DelimitedFiles
using LinearAlgebra
using Plots
using CSV

# ARQUIVOS DE FUNÇÕES E OUTROS, PRESENTES NO DIRETÓRIO DO PROGRAMA
include(pwd() * "\\LeakyRidgeESN.jl")
include(pwd() * "\\MeasuresESN.jl")

println("\nESN > Echo State Neural Network")

# DEFINE O SISTEMA A SER ANALISADO
dinsystem = DelimitedFiles.readdlm("FourierImag.txt", ',')
dinsystem = reshape(dinsystem[:,1], (size(dinsystem)[1], 1))
print(size(dinsystem))
#println(dinsystem)
# DEFINE O QUE É INPUT E O QUE É OUTPUT - ATRAVÉS DAS DIMENSÕES DO SISTEMA
u = dinsystem
y = dinsystem
# TOTAL DE DADOS PRESENTES PARA A ANÁLISE (TAU)
total_data = size(dinsystem)[1]
# DIMENSÕES DO SISTEMA
input_dim = size(u)[2]
output_dim = size(y)[2]

# DEFINE A SEED DO ALGORITMO PSEUDO-ALEATÓRIO
Random.seed!(42)
# OBSERVAÇÕES PARA TREINAMENTO
training_length = 150::Int64
# OBSERVAÇÕES PARA TESTE
testing_length = 3::Int64
# TRANSIENTE
transient = 40::Int64
# QUANTIDADE DE JANELAS
window = 350::Int64

# NÚMERO DE UNIDADES DE PROCESSAMENTO (NEURÔNIOS)
n_neurons = 600::Int64
# RAIO ESPECTRAL
spectral_radius = 1.25::Float64
# TAXA DE VAZAMENTO
leaking_rate = 0.3::Float64
# ESPARCIDADE
sparsity = 0.4::Float64
# MENSAGENS
messages = true::Bool

println("ESN > Total input dimensions: ", size(u), " [T x K]")
println("ESN > Total output dimensions: ", size(y), " [T x L]")
println("ESN > Training length: ", training_length)
println("ESN > Testing length: ", testing_length)
println("ESN > Reservoir's processing units: ", n_neurons, " neurons")
println("ESN > Reservoir's spectral radius: ", spectral_radius)

# DEFINIMOS UMA MATRIZ DE PESOS DE ENTRADA
w_input = input_weights(input_dim = input_dim, 
                        n_neurons = n_neurons,
                        input_scale = 1.0,
                        messages = false)

# DEFINIMOS UMA MATRIZ DE PESOS DE RESERVATÓRIO (CONEXÕES ENTRE NEURÔNIOS) 
w_reservoir = reservoir_weights(n_neurons = n_neurons, 
                                spectral_radius = spectral_radius,
                                sparsity = sparsity,
                                messages = false)

# MATRIZES DE COMPARAÇÃO (PREDITO VS REAL)
test_matrix_x = zeros(window, 1)
#test_matrix_y = zeros(window, 1)
pred_matrix_x = zeros(window, 1)
#pred_matrix_y = zeros(window, 1)

# PARA CADA JANELA
for t in 1:window

    println("\nESN > Step: ", t)
    # ENTRADA PARA CADA RODADA DE TREINAMENTO
    u_train = u[t : t + training_length - 1, :]
    # OUTPUT DO TRAINAMENTO
    y_train = y[t + transient + 1 : t + training_length, :]
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
    
    println("ESN > Medidas de erro - Trainning")
    y_hat_train = transpose(w_output * extended_states)
    mean_squared_error(data = y_train, pred_data = y_hat_train)
    normalized_mean_squared_error(data = y_train, pred_data = y_hat_train)
    mean_absolute_percentage_error(data = y_train, pred_data = y_hat_train)

    # PREDICTION
    local y_hat = predict(w_output = w_output, 
                          w_input = w_input, 
                          w_reservoir = w_reservoir,
                          u = u[t + training_length, :], 
                          states = last_state, 
                          output_dim = output_dim, 
                          to_predict = testing_length, 
                          leaking_rate = leaking_rate, 
                          messages = false)
    
    test_matrix_x[t, :] .= u[t + training_length + testing_length, 1]
    #test_matrix_y[t, :] .= u[t + training_length + testing_length, 2]
    pred_matrix_x[t, :] .= y_hat[testing_length, 1]
    #pred_matrix_y[t, :] .= y_hat[testing_length, 2]
end   

# MEDIDAS DE ERRO DA VALIDAÇÃO
println("\nESN > Medidas de erro - Validação")
mean_squared_error(data = test_matrix_x, pred_data = pred_matrix_x)
normalized_mean_squared_error(data = test_matrix_x, pred_data = pred_matrix_x)
mean_absolute_percentage_error(data = test_matrix_x, pred_data = pred_matrix_x)

# PLOT
plot(test_matrix_x, label="System", color="blue", legend=:topleft)
plot!(pred_matrix_x, label="Prediction", color="red", legend=:topleft)

#scatter(pred_matrix_x, test_matrix_x, label="Comparison", color="blue", legend=:topleft)
#plot!(test_matrix_x, test_matrix_x, label="45 degree", color="black", legend=:topleft)