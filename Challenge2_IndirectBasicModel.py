import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def coupled_system(y, t, I, kia, Kia, Fa, kfaa, Kfaa, kab, Kab, Fb, kfbb, Kfbb, kac, Kac, kbc, Kbc):
    A, B, C = y

    if t < 0.2 * total_time:
        input_signal = 1
    elif t < 0.4 * total_time:
        input_signal = 2
    elif t < 0.6 * total_time:
        input_signal = 1
    elif t < 0.8 * total_time:
        input_signal = 2
    else:
        input_signal = 1
    dA_dt =  input_signal * kia * ((1 - A) / ((1 - A) + Kia)) - Fa * kfaa * (A / (A + Kfaa))
    dB_dt = A * kab * ((1 - B) / ((1 - B) + Kab)) - Fb * kfbb * (B / (B + Kfbb))
    dC_dt = A * kac * ((1 - C) / ((1 - C) + Kac)) - B * kbc * (C / (C + Kbc))
    
    return [dA_dt, dB_dt, dC_dt]

def integrate_system(y0, time_steps, params):
    global total_time
    total_time = time_steps[-1]

    result = odeint(
        lambda y, t: coupled_system(y, t, 
                                     params['I'], params['kia'], params['Kia'], 
                                     params['Fa'], params['kfaa'], params['Kfaa'], 
                                     params['kab'], params['Kab'], 
                                     params['Fb'], params['kfbb'], params['Kfbb'], 
                                     params['kac'], params['Kac'], 
                                     params['kbc'], params['Kbc']),
        y0, time_steps
    )
    return result

def plot_C(time_steps, results):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, results[:, 2], label='Concentration of C')
    plt.xlabel('Time')
    plt.ylabel('Concentration of C')
    plt.title('Concentration of C over Time')
    plt.legend()
    plt.grid()
    plt.show()

def parameter_sweep(y0, time_steps, param_sets):
    results = []
    for params in param_sets:
        result = integrate_system(y0, time_steps, params)
        results.append(result)
    return results

def main():
    # y0 = [.1, .1, .1]
    # time_steps = np.linspace(0, 100, 1000)
    y0 = [.5, 10, 20]
    time_steps = np.linspace(0, 1000, 1000)


    # param_sets = [
    #     {'I': 1, 'kia': 5, 'Kia': 1, 'Fa': 1, 'kfaa': 1, 'Kfaa': 1, 'kab': 0.1, 'Kab': 0.001, 'Fb': 0.1, 'kfbb': 1, 'Kfbb': 10, 'kac': 1, 'Kac': 0.1, 'kbc': 1, 'Kbc': 10},
    #     ]
    
    param_sets = [
        {'I': 1, 'kia': 10, 'Kia': 10, 'Fa': 10, 'kfaa': 10, 'Kfaa': 10, 'kab': 1, 'Kab': 0.01, 'Fb': 1, 'kfbb': 10, 'Kfbb': 100, 'kac': 10, 'Kac': 1, 'kbc': 10, 'Kbc': 100},
        {'I': 1, 'kia': 10, 'Kia': 1, 'Fa': 10, 'kfaa': 10, 'Kfaa': 10, 'kab': 1, 'Kab': 0.01, 'Fb': 1, 'kfbb': 10, 'Kfbb': 100, 'kac': 10, 'Kac': 1, 'kbc': 10, 'Kbc': 100},
        ]
    
    all_results = parameter_sweep(y0, time_steps, param_sets)
    
    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:cyan', 'magenta', 'green', 'pink']

    for i, result in enumerate(all_results):
        plt.plot(time_steps, result[:, 2], label=f'Sweep Param Set {i+1}', color=colors[i % len(colors)])
    
    plt.axvline(x=0.2*total_time, color='limegreen', label='Signal On', linestyle='--')
    plt.axvline(x=0.4*total_time, color='red', label='Signal Off', linestyle='--')
    plt.axvline(x=0.6*total_time, color='limegreen', linestyle='--')
    plt.axvline(x=0.8*total_time, color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration of C')
    plt.title('Concentration of C over Time')
    plt.legend()
    plt.grid()
    plt.savefig('results/indirect_basic_model.pdf')



if __name__ == "__main__":
    main()