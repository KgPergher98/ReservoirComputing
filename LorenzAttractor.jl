#=
 LORENZ ATTRACTOR

 FROM https://github.com/pjpmarques/Julia-Modeling-the-World/blob/master/Lorenz%20Attractor.ipynb
=#

#using PyPlot
using Plots
using ODE
#using DifferentialEquations

# FUNÇÃO A SER UTILIZADA NO DECORRER DO PROJETO
function lorenzAttractor(;r0 = [0.1; 0.0; 0.0], dt = 0.1, tf = 100.00, rshape = true, n_data = 10000)

    function lorenz(t, r)

        (x, y, z) = r
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3.0
        dx_dt = sigma * (y - x)
        dy_dt = (x * (rho - z)) - y
        dz_dt = (x * y) - (beta * z)
    
        return [dx_dt; dy_dt; dz_dt]
    end
    
    t = collect(dt:dt:tf)

    (t, pos) = ode23(lorenz, r0, t)

    #print(pos)
    x = map(v -> v[1], pos)
    y = map(v -> v[2], pos)
    z = map(v -> v[3], pos)

    x = x[1:n_data]
    y = y[1:n_data]
    z = z[1:n_data]

    if rshape
        x = reshape(x, size(x)[1], 1)
        y = reshape(y, size(y)[1], 1)
        z = reshape(z, size(z)[1], 1)
    end
    
    return x, y, z
end

# PARA TESTAR O MÓDULO
#x, y, z = lorenzAttractor(r0 = [0.1; 0.0; 0.0], dt = 0.01, tf = 100.00, rshape = true, n_data = 2000)
#print(size(x))

#plot(x, y, z, plotstyle="points")
#plot(x, y, z, plotstyle="points")