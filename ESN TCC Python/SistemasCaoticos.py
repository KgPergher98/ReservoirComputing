# Principais sistemas caóticos a serem analisados

import numpy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import signalz

def reshaping_data(data):
        if data.ndim < 2:
            data = numpy.reshape(data, (len(data), 1))
        return data

def henon_map(x0=0, y0=0, n_observations=100):
    """ Implementa o mapa de Hénon, um sistema caótico dado por:

        x(n+1) = 1 - ax^2(n) + y(n)
        y(n+1) = bx(n)

        Onde a = 1.4 e b = 0.3 dão origem a uma órbita com atrator
    """
    # Dá origem a quantidade de pontos a serem analisados
    data = numpy.zeros((n_observations,2))
    # Condição inicial do problema (xo, yo)
    data[0,0] = x0
    data[0,1] = y0
    # Constantes da trajetória de Hénon
    a = 1.4
    b = 0.3
    # Criação da trajetória
    for point in range(1, n_observations):
        data[point,0] = 1 - (a*pow(data[point-1,0], 2)) + data[point-1,1]
        data[point,1] = b*data[point-1,0]
    #return numpy.hstack([reshaping_data(x), reshaping_data(y)])
    return data

def lorenz_attractor(state0=(1,1,1), rho=28.0, sigma=10.0, beta=8/3, n_observations=4000, final_t=40):

    def equation(state, t):
        x, y, z = state
        return sigma*(y-x), x*(rho-z)-y, x*y-beta*z

    t = numpy.arange(0,final_t, final_t/n_observations)
    return odeint(equation, state0, t)

def gaussian_noise(mean = 0, std = 1, n_observations = 1000, dim = 1):
    return numpy.random.normal(mean, std, (n_observations, dim))

def student_noise(df = 4, n_observations = 1000, dim = 1):
    return numpy.random.standard_t(df, (n_observations, dim))

def mackey_glass(n_observations, beta=0.2, theta=0.8, gama=0.9, tau=23, n=10, p0=0.1):
    x = signalz.mackey_glass(n_observations, a=beta, b=theta, c=gama, d=tau, e=n, initial=p0)
    return numpy.reshape(x, (x.shape[0],1))


if __name__ == "__main__":

    # Gaussian Noise
    gn = gaussian_noise(mean = 0, std = 1, n_observations = 1000000, dim = 1)
    numpy.savetxt("gaussianNoise1D.txt", gn, delimiter = ",")
    gn = gaussian_noise(mean = 0, std = 1, n_observations = 1000000, dim = 2)
    numpy.savetxt("gaussianNoise2D.txt", gn, delimiter = ",")
    gn = gaussian_noise(mean = 0, std = 1, n_observations = 1000000, dim = 3)
    numpy.savetxt("gaussianNoise3D.txt", gn, delimiter = ",")

    # Lorenz Attractor
    """
    l_att = lorenz_attractor(state0 = (1,1,1), rho = 28, sigma = 10, 
                             beta = 8/3, n_observations = 1000000, final_t = 10000)
    print(type(l_att))
    print(l_att.shape)
    numpy.savetxt("lorenzAttractor.txt", l_att, delimiter = ",")
    """

    # T STUDENT
    noise = student_noise(df = 4, n_observations = 1000000, dim = 1)
    numpy.savetxt("studentNoise1D.txt", noise, delimiter = ",")
    noise = student_noise(df = 4, n_observations = 1000000, dim = 2)
    numpy.savetxt("studentNoise2D.txt", noise, delimiter = ",")
    noise = student_noise(df = 4, n_observations = 1000000, dim = 3)
    numpy.savetxt("studentNoise3D.txt", noise, delimiter = ",")

    # Henon Map
    """
    h_map = henon_map(x0 = 0, y0 = 0, n_observations = 1000000)
    print(type(h_map))
    print(h_map.shape)
    numpy.savetxt("henonMap.txt", h_map, delimiter = ",")
    """

    # Mackey Glass 17
    """
    mack_glass = mackey_glass(n_observations = 1000000, beta = 0.2, theta = 0.8, gama = 0.9, tau = 17, 
                              n = 10, p0 = 0.1)
    print(type(mack_glass))
    print(mack_glass.shape)
    numpy.savetxt("mackeyGlass17.txt", mack_glass)
    """

    # Mackey Glass 22
    """
    mack_glass = mackey_glass(n_observations = 1000000, beta = 0.2, theta = 0.8, gama = 0.9, tau = 22, 
                              n = 10, p0 = 0.1)
    print(type(mack_glass))
    print(mack_glass.shape)
    numpy.savetxt("mackeyGlass22.txt", mack_glass)
    """

    # Plot Lorenz Attractor
    """
    plt.figure()
    plt.gca(projection="3d")
    plt.plot(l_att[:, 0], l_att[:, 1], l_att[:, 2])
    plt.draw()
    plt.show()
    """

    # Plot Henon Map
    """
    plt.figure()
    plt.scatter(h_map[:,0], h_map[:,1])
    plt.show()
    """

    # Plot Mackey Glass
    """
    plt.figure()
    plt.plot(numpy.arange(0,1000000), mack_glass)
    plt.show()
    """