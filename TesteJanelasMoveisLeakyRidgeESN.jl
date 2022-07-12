#=
    TESTAGEM DO USO DE ECHO STATE NEURAL NETWORKS COM LEAKING RATE 
    TESTAGEM COM O USO DE SISTEMAS CAÓTICOS PADRÃO
    MACKEY GLASS, MAPA DE HÉNON E ATRATOR DE LORENZ 

    CRIADO EM 03/07/2022
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

println("\nESN > Echo State Neural Network")

# DEFINE O SISTEMA A SER ANALISADO
dinsystem = transpose(readdlm("henonMap.txt", ','))
println(size(dinsystem))
# TOTAL DE DADOS PRESENTES PARA A ANÁLISE (TAU)
total_data = size(dinsystem)[2]
# DEFINE O QUE É INPUT E O QUE É OUTPUT - ATRAVÉS DAS DIMENSÕES DO SISTEMA
#x1 = reshape(dinsystem[1, :], (total_data, 1))
#x2 = reshape(dinsystem[2, :], (total_data, 1))
#u = x2
#y = x2
u = transpose(dinsystem)
y = transpose(dinsystem)
# DIMENSÕES DO SISTEMA
input_dim = size(u)[2]
output_dim = size(y)[2]

# DEFINE A SEED DO ALGORITMO PSEUDO-ALEATÓRIO
Random.seed!(42)
# OBSERVAÇÕES PARA TREINAMENTO
training_length = 200::Int64
# OBSERVAÇÕES PARA TESTE
testing_length = 5::Int64
# TRANSIENTE
transient = 10::Int64
# QUANTIDADE DE JANELAS
window = 200::Int64

# NÚMERO DE UNIDADES DE PROCESSAMENTO (NEURÔNIOS)
n_neurons = 500::Int64
# RAIO ESPECTRAL
spectral_radius = 1.25::Float64
# TAXA DE VAZAMENTO
leaking_rate = 0.3::Float64
# ESPARCIDADE
sparsity = 0.3::Float64
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
                        messages = true)

# DEFINIMOS UMA MATRIZ DE PESOS DE RESERVATÓRIO (CONEXÕES ENTRE NEURÔNIOS) 
w_reservoir = reservoir_weights(n_neurons = n_neurons, 
                                spectral_radius = spectral_radius,
                                sparsity = sparsity,
                                messages = true)

# MATRIZES DE COMPARAÇÃO (PREDITO VS REAL)
test_matrix_x = zeros(window, 1)
test_matrix_y = zeros(window, 1)
pred_matrix_x = zeros(window, 1)
pred_matrix_y = zeros(window, 1)

# PARA CADA JANELA
for t in 1:window

    println("\nESN > Analysis: ", t)
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
                                                       messages = true)
    
    # OUTPUT ATRAVÉS DA REGRESSÃO DE TIKHONOV
    local w_output = tikhonov_regression(extended_states = extended_states, 
                                         output = y_train, 
                                         messages = true)
    
    # PREDICTION
    local y_hat = predict(w_output = w_output, 
                          w_input = w_input, 
                          w_reservoir = w_reservoir,
                          u = u[t + training_length, :], 
                          states = last_state, 
                          output_dim = output_dim, 
                          to_predict = testing_length, 
                          leaking_rate = leaking_rate, 
                          messages = true)
    
    println(u[t + training_length, :])
    println(u[t + training_length + testing_length, :])
    println(y_hat)
    println(size(y_hat))
    test_matrix_x[t, :] .= u[t + training_length + testing_length, 1]
    test_matrix_y[t, :] .= u[t + training_length + testing_length, 2]
    pred_matrix_x[t, :] .= y_hat[testing_length, 1]
    pred_matrix_y[t, :] .= y_hat[testing_length, 2]

end   

#plot(test_matrix); plot!(pred_matrix)
scatter(test_matrix_x, test_matrix_y, label="System", color="blue")
scatter!(pred_matrix_x, pred_matrix_y, label="Prediction", color="red")