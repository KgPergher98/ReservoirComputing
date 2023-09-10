#=
    ECHO STATE NEURAL NETWORK

    REDE DE ECHO "TRADICIONAL" DA LITERATURA, COM USO DE TAXA DE VAZAMENTO
    COM USO DE PESOS DE ENTRADA (INPUT), SAÍDA (OUTPUT), E DE RESERVATÓRIO. 
    RESOLUÇÃO DE MATRIZ OUTPUT DO SISTEMA POR MEIO DE REGRESSÃO DE RIDGE

    CRIADO POR KEVIN PERGHER
    PORTO ALEGRE, 12 DE JUNHO DE 2022
=#  

# PRINCIPAIS BIBLIOTECAS A SEREM UTILIZADAS NA EXECUÇÃO DESTE PROGRAMA
using Random
using DelimitedFiles
using LinearAlgebra
using Plots

# FUNÇÃO RESPONSÁVEL PELA CRIAÇÃO DO RESERVATÓRIO (MATRIZ DE PESOS ENTRE AS UNIDADES DE PROCESSAMENTO)
function reservoir_weights(;n_neurons::Int64 = 2, 
                           spectral_radius::Float64 = 1.0,
                           sparsity::Float64 = 0.0,
                           messages::Bool = false)
    # CRIA UMA MATRIZ DE PESOS DE RESERVATORIO DE DIMENSÃO N_NEURONS X N_NEURONS
    # OS VALORES DA MATRIZ SÃO POSITIVOS EM [0, 1)
    w_reservoir = rand(Float64, (n_neurons, n_neurons)) .- 0.5
    # MAIOR AUTOVALOR (EIGENVALUE) ABSOLUTO (REAL) DA MATRIZ DE PESOS
    max_eigenvalue = maximum(abs.(eigvals(w_reservoir)))
    # DEFINIMAOS O RAIO ESPECTRAL DO RESERVATÓRIO COMO O PARÂMETRO DEFINIDO
    w_reservoir = w_reservoir .* (spectral_radius / max_eigenvalue)
    # APLICAÇÃO DE "ESPARCIDADE" NA MATRIZ DE PESOS DO RESERVATÓRIO
    if sparsity > 0
        # ESCOLHEMOS PSEUDO-RANDOMICAMENTE CONEXÕES A SEREM DELIBERADAMENTE ZERADAS
        # A PORCENTAGEM DE CONEXÕES "ZERADAS" EQUIVALE A SPARSITY
        w_reservoir[rand(Float64, (n_neurons, n_neurons)) .< sparsity] .= 0
    end
    # RETORNA OS PESOS DO RESERVATÓRIO
    if messages
        println("ESN > Reservoir's weight size: ", size(w_reservoir), " [N x N]")
    end
    return w_reservoir
end

# FUNÇÃO RESPONSÁVEL PELA CRIAÇÃO DA MATRIZ DE PESOS DE ENTRADA
function input_weights(;input_dim::Int64 = 1,
                       n_neurons::Int64 = 2,
                       input_scale::Float64 = 1.0,
                       messages::Bool = false)
    # CRIAMOS A MATRIZ DE PESOS DE ENTRADA 
    # CONEXÃO FIXA ENTRE u(t) [INPUT] E OS NEURÔNIOS NO RESERVATÓRIO
    # A ESCALA DO RESERVATÓRIO É UM HIPER-PARÂMETRO DA REDE 
    w_input = (rand(Float64, (n_neurons, 1 + input_dim)) .- 0.5) .* input_scale
    # RETORNA A MATRIZ DE PESOS DE ENTRADA
    if messages
        println("ESN > Input's weight size: ", size(w_input), " [N x (K + 1)]")
    end
    return w_input
end

# APLICA A REGRA DE ATUALIZAÇÃO SIMLPES DOS ESTADOS X(n)
function simple_update(;win, wres, u, states)
    # REGRA DE ATUALIZAÇÃO SIMPLES, UTILIZA OS PESOS DE ENTRADA E DO RESERVATÓRIO
    # VÁLIDA PARA O CASO DE ATUALIZAÇÃO SIMPLES COM/SEM "TAXA DE VAZAMENTO"
    # X(n + 1) = tanh( (Winput * u(n + 1)) + (Wreservoir * X(n)) )
    return tanh.((win * u) + (wres * states))
end

# "COLHEITA DOS ESTADOS", A MATRIZ DE ESTADOS É CRIADA CONFORME A REGRA DE ATUALIZAÇÃO ESCOLHIDA
function harvest_states(;w_input, w_reservoir, u, 
                        n_neurons::Int64 = 100, input_dim::Int64 = 1,
                        training_length::Int64 = 100, 
                        transient::Int64 = 1, 
                        leaking_rate::Float64 = 0.3,
                        messages::Bool = false)
    # PROCESSAMENTO DA MATRIZ DE ESTADOS X (EM FASE DE TREINAMENTO, PRÉ-REGRESSÃO)
    # W_INPUT := MATRIZ DE PESOS DE ENTRADA, [NUMERO_NEURONIOS x TAMANHO_ENTRADA]
    # W_RESERVOIR := MATRIZ DE PESOS DE CONEXÕES DO RESERVATÓRIO (ENTRE NEURÔNIOS), [NUMERO_NEURONIOS x NUMERO_NEURONIOS]
    # U := VALORES DE SINAL DE ENTRADA (INPUT)
    # Y := VALORES DE SINAL DE SAÍDA (OUTPUT)
    # X := MATRIZ DE ESTADOS

    # CRIAÇÃO DA MATRIZ DE ESTADOS "EXTENDIDOS", POR DEFAULT OS ESTADOS SÃO ZERADOS
    x = zeros(1 + input_dim + n_neurons, training_length - transient)
    # VARIÁVEL AUXILIAR, EQUIVALE AOS ESTADOS NO PERÍODO n
    states = zeros(n_neurons, 1)

    for t in 1:training_length
        # CONCATENAÇÃO MATRICIAL DE [1] (RELACIONADO AO "BIAS" NO PARADIGMA RNN) E u(n + 1)
        u_t = vcat([1], u[t,:])
        # ATUALIZAÇÃO DOS ESTADOS
        states = ((1 - leaking_rate) .* states) .+ (leaking_rate .* simple_update(win = w_input, wres = w_reservoir, u = u_t, states = states)) 
        if t > transient
            # VALORES INFERIORES AO TRANSIENTE SÃO IGNORADOS, QUESTÃO DE ESTABILIDADE DA REDE
            x[:, t - transient] = vcat(u_t, states) 
        end
    end

    if messages
        println("ESN > Extended states' size: ", size(x), " [(N + K + 1) x (n_max - transient)]")
    end

    # RETORNA A MATRIX DE ESTADOS (PARA CÁLCULO DA MATRIZ DE OUTPUT) E O ÚLTIMO ESTADO PARA "PREDICTION"
    return x, states
end

# ESTIMA A MATRIZ "IDEIAL" DE SAÍDA (W_out) ATRAVÉS DA REGRESSÃO DE TIKHONOV (RIDGE) 
function tikhonov_regression(;extended_states, 
                             output, 
                             alpha::Float64 = 1e-8,
                             messages::Bool = false)
    # AGRADECER AO PROF. MANTAS LUKOSEVICIUS (https://mantas.info/) QUE APARENTEMENTE DESCOBRIU QUE A SEGUINTE IMPLEMENTAÇÃO
    # DA REGRESSÃO DE TIKHONOV É "MELHOR" QUE SIMPLESMENTE REPLICAR A EQUAÇÃO 
    w_out = transpose((extended_states * transpose(extended_states) + alpha * I) \ (extended_states * output))

    if messages
        println("ESN > Output's matrix: ", size(w_out), " [L x (N + K + 1)]")
    end

    return w_out
end

# REALIZA A PREVISÃO DA SÉRIE TEMPORAL n PASSOS A FRENTE
function predict(;w_reservoir, w_input, w_output, 
                 u, states, output_dim::Int64 = 1, 
                 to_predict::Int64 = 1, leaking_rate::Float64 = 0.3,
                 messages::Bool = false)
    # MATRIZ DE "FORECATSING"
    prediction = zeros(to_predict, output_dim)

    for t in 1:to_predict
        # CONCATENAÇÃO MATRICIAL DE [1] E u(n + 1)
        u_t = vcat([1], u)
        # ATUALIZAÇÃO DOS ESTADOS CONFORME REGRA ESTABELECIDA
        states = ((1 - leaking_rate) .* states) + (leaking_rate .* simple_update(win = w_input, wres = w_reservoir, 
                                                                                 u = u_t, states = states))
        # O PREDITOR DE y É OBTIDO PELA APLICAÇÃO DA MATRIZ W_out SOBRE [1, u(n + 1), x(n + 1)]
        y_hat = w_output * vcat([1], u, states)
        prediction[t, :] = y_hat
        # A PRÓXIMA ENTRA É A PRÓPRIA PREVISÃO (GENERATIVE MODE/ "FREE RUNNING")
        u = y_hat
    end

    if messages
        println("ESN > Prediction's size: ", size(prediction), " [n_testing x L]")
    end

    return prediction
end