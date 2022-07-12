#=
    TESTES PARA A REDE ESN COM FEEDBACK
=#

# PRINCIPAIS BIBLIOTECAS A SEREM UTILIZADAS NA EXECUÇÃO DESTE PROGRAMA
using Random
using DelimitedFiles
using LinearAlgebra
using Plots

# ARQUIVOS DE FUNÇÕES E OUTROS, PRESENTES NO DIRETÓRIO DO PROGRAMA
include(pwd() * "\\MackeyGlass.jl")
include(pwd() * "\\LorenzAttractor.jl")
include(pwd() * "\\HenonMap.jl")
include(pwd() * "\\FeedbackRidgeESN.jl")

# SISTEMA A SER ANALISADO
case = "HenonMap"
# DEFINE A SEED DO ALGORITMO PSEUDO-ALEATÓRIO
Random.seed!(42)
# OBSERVAÇÕES PARA TREINAMENTO
training_length = 3000::Int64
# OBSERVAÇÕES PARA TESTE
testing_length = 100::Int64
# TRANSIENTE
init_length = 100::Int64
# NÚMERO DE UNIDADES DE PROCESSAMENTO (NEURÔNIOS)
n_neurons = 3000::Int64
# RAIO ESPECTRAL
spectral_radius = 1.25::Float64
# TAXA DE VAZAMENTO
leaking_rate = 0.3::Float64
# ESPARCIDADE
sparsity = 0.3::Float64
# MENSAGENS
messages = true::Bool

# DADOS RELATIVOS AOS SISTEMAS DE INTERESSE
# DISCLAIMER, APESAR DE "u" E "y" POR VEZES SE CONFUNDIREM,
#             ELES SÃO ESSENCIALMENTE DIFERENTES E É BOM QUE ISTO ESTEJA NO CÓDIGO
#             u := INPUT, COM K DIMENSÕES E T OBSERVAÇÕES
#             y := OUTPUT, COM L DIMENSÕES E T OBSERVAÇÕES
if case == "MantasMackeyGlass"
    println("Analisando Mackey-Glass, dados prof. Mantas")
    y = readdlm("mackeyGlassMantas.txt")
    u = y
elseif case == "MackeyGlass"
    println("Mackey-Glass")
    x, y = MGGenerator(sample_n = training_length + testing_length - 1, 
                       x0 = 1.2, deltat = 0.1, 
                       tau = 17, a = 0.2, b = 0.1)
    u = y
elseif case == "HenonMap"
    println("Hénon Map")
    x, y = henonMap(r0 = [0.0; 0.0], a = 1.4, b = 0.3, 
                    n_iter = training_length + testing_length)
    u = y
elseif case == "LorenzAttractor"
    println("Lorenz Attractor")
    x, y, z = lorenzAttractor(r0 = [0.1; 0.0; 0.0], dt = 0.01, tf = 100.00, 
                              rshape = true, n_data = training_length + testing_length)
    u = y
end

println("\nESN > Echo State Neural Network")
println("ESN > Total input dimensions: ", size(u), " [T x K]")
println("ESN > Total output dimensions: ", size(y), " [T x L]")
println("ESN > Training length: ", training_length)
println("ESN > Testing length: ", testing_length)
println("ESN > Reservoir's processing units: ", n_neurons, " neurons")
println("ESN > Reservoir's spectral radius: ", spectral_radius)

input_dim = 1
output_dim = 1

w_input = input_weights(input_dim = input_dim, 
                        n_neurons = n_neurons,
                        input_scale = 1.0,
                        messages = true)

w_feedback = feedback_weights(output_dim = output_dim,
                              n_neurons = n_neurons,
                              feedback_scale = 1.0,
                              messages = true)

w_reservoir = reservoir_weights(n_neurons = n_neurons, 
                                spectral_radius = spectral_radius,
                                sparsity = sparsity,
                                messages = true)

extended_states, last_state = feedback_harvest_states(w_input = w_input, 
                                                      w_reservoir = w_reservoir,
                                                      w_back = w_feedback, 
                                                      u = u, y = y, n_neurons = n_neurons, 
                                                      input_dim = input_dim,
                                                      training_length = training_length,
                                                      transient = init_length,
                                                      messages = true)

w_output = tikhonov_regression(extended_states = extended_states, 
                               output = y[init_length + 2 : training_length + 1, :], 
                               messages = true)

y_hat = feedback_predict(w_output = w_output, w_input = w_input, w_reservoir = w_reservoir,
                         w_back = w_feedback, y = y[training_length, :],
                         u = u[training_length + 1, :], states = last_state, output_dim = output_dim, 
                         to_predict = testing_length - 1, messages = true)

y_testing = y[training_length + 2 : training_length + testing_length, :]
#println(y_testing)
#println(y_hat)
#println(u[training_length + 1, :])
plot(y_testing); plot!(y_hat)