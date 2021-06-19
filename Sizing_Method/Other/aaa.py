# author: Bao Li # 
# Georgia Institute of Technology #
import numpy as np
import matplotlib.pylab as plt


def p1():
    end = 2
    h = [0.5, 0.05, 0.005]
    x_list = [0.5, 1.0, 1.5, 2.0]
    error = np.zeros([3, 4])
    h_label = ['h=0.5', 'h=0.05', 'h=0.005']
    for i in range(len(h)):
        x = 0
        y = 3
        n = int(end/h[i])
        for j in range(n+1):
            dydx = -1.2*y + 7*np.exp(-0.3*x)
            y1 = y + dydx*h[i]
            for k in range(len(x_list)):
                if abs(x - x_list[k]) < 0.0001:
                    y_ture = 70 / 9 * np.exp(-0.3 * x) - 43 / 9 * np.exp(-1.2 * x)
                    error[i, k] = abs((y_ture-y)/y_ture)

            x = x + h[i]
            y = y1
    print(error)
    plt.figure(figsize=(8, 6))

    for i in range(3):
        plt.plot(x_list, np.log10(error[i, :]), 'o-', label=h_label[i])

    plt.xlabel('x')
    plt.ylabel('log10(ture error)')
    plt.title('Comparison of Errors')
    plt.legend(loc=0)
    plt.grid()
    plt.show()


def p2():
    end = 2
    h = 0.2
    x = 0
    y = 0
    dydx = 1
    n = int(end/h)+1
    error = np.zeros(n+1)
    y_ture = np.zeros(n+1)
    y = np.zeros(n+2)
    y[0] = 0

    for i in range(n+1):
        x_half = x + h / 2
        y_half = y[i] + dydx*h/2
        dydx_half = y_half*(-2*x_half+1/x_half)
        y[i+1] = y[i] + dydx_half*h
        x2 = x + h
        y_ture[i] = x*np.exp(-x**2)
        if i != 0:
            error[i] = abs((y_ture[i]-y[i])/y_ture[i])

        x = x2
        dydx = y[i+1] * (-2 * x + 1 / x)

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0,2,n),  np.log10(error[1:]),'o-')
    plt.xlabel('x')
    plt.ylabel('log10(ture error)')
    plt.title('Comparison of Errors')
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, 2, n-1), y_ture[1:-1], 'o-', label='true')
    plt.plot(np.linspace(0, 2, n-1), y[1:-2], 'o-', label='approx')
    print(y)
    print(y_ture)
    print(error)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Values')
    plt.legend(loc=0)
    plt.grid()
    plt.show()