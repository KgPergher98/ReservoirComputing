#=
    MAPA DE HÉNON BIDIMENSIONAL

    DESENVOLVIDO POR KEVIN PERGHER
    PORTO ALEGRE, 19/06/2022
=#

using Plots
#using PyPlot

# FUNÇÃO GERADORA DA TRAJETÓRIA NO MAPA DE HÉNON
function henonMap(;r0 = [0.0; 0.0], a = 1.4, b = 0.3, n_iter = 1000)
    # CRIA OS VETORES EM SEPARADO
    x = zeros(n_iter, 1)
    y = zeros(n_iter, 1)
    # ATRIBUI OS VALORES INICIAIS
    x[1, 1] = r0[1]
    y[1, 1] = r0[2]
    # REALIZA AS ITERAÇÕES
    for t in 2:n_iter
        x[t, 1] = 1 - (a * (x[t-1, 1]^2)) + y[t-1, 1]
        y[t, 1] = b * x[t-1, 1]
    end
    # RETORNA OS VALORES
    return x, y
end

# UM PEQUENO TESTE DO MÓDULO
#x, y = henonMap()
#print(size(x))