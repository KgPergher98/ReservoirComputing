#=
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" Julia.
from https://mantas.info/code/simple_esn/
(c) 2020 Mantas LukoÅ¡eviÄius
Distributed under MIT license https://opensource.org/licenses/MIT
=#

using DelimitedFiles
import Random
using Plots
using LinearAlgebra

# DEFINE RANDOM SEED 
Random.seed!(42)
println("This was proposed by Mantas")
# DEFINIÇÃO DOS DADOS DE TREINAMENTO, trainLen = 2000
trainLen = 20
# DEFINIÇÃO DOS DADOS DE TESTE, testLen = 2000
testLen = 20
# DEFINIÇÃO DO TRANSIENTE, initLen = 100
initLen = 1
# LEITURA DOS DADOS
data = readdlm("mackeyGlassMantas.txt")


# generate the ESN reservoir
inSize = outSize = 1
resSize = 10
a = 0.3 # leaking rate

# CRIAÇÃO DOS PESOS DE INPUT
Win = (rand(resSize, 1+inSize) .- 0.5) .* 1
# CRIAÇÃO DO RESERVATÓRIO
W = rand(resSize, resSize) .- 0.5 
# NORMALIZATION BY SPECTRAL RADIUS
rhoW = maximum(abs.(eigvals(W)))
W .*= (1.25 / rhoW)

# allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize, trainLen-initLen)
# set the corresponding target matrix directly
Yt = transpose(data[initLen+2:trainLen+1]) 

# run the reservoir with the data and collect X
x = zeros(resSize, 1)
for t = 1:trainLen
    u = data[t]
    global x = (1-a).*x .+ a.*tanh.( Win*[1;u] .+ W*x ) 
    if t > initLen
        X[:,t-initLen] = [1;u;x]
    end
end

# train the output by ridge regression
reg = 1e-8  # regularization coefficient
# direct equations from texts:
#X_T = transpose(X) 
#Wout = Yt*X_T * inv(X*X_T + reg*I)
# using Julia backslash solver:
Wout = transpose((X*transpose(X) + reg*I) \ (X*transpose(Yt)))


# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = zeros(outSize, testLen)
u = data[trainLen+1]
for t = 1:testLen 
	global x = (1-a).*x .+ a.*tanh.( Win*[1;u] .+ W*x )
	y = Wout*[1;u;x]
	Y[:,t] = y
	# generative mode:
	global u = y
	# this would be a predictive mode:
	#global u = data[trainLen+t+1]
end
println(Y)
print("end")
#=
# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( abs2.( data[trainLen+2:trainLen+errorLen+1] .- 
    Y[1,1:errorLen] ) ) / errorLen
println("MSE = $mse")

# plot some signals
p2 = plot(data[trainLen+2:trainLen+testLen+1], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
title!("Target and generated signals y(n) starting at n=0")

p3 = plot( transpose(X[1:20,1:200]), leg = false )
title!("Some reservoir activations x(n)")

p4 = bar(transpose(Wout), leg = false)
title!("Output weights Wout")

# display all 4 plots
plot(p1, p2, p3, p4, size=(1200,800))
=#