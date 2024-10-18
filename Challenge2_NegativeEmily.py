import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def coupled_system(y, t, I, kia, Kia, Fa, kfaa, Kfaa, kcb, Kcb, Fb, kfbb, Kfbb, kac, Kac, kbc, Kbc):
    A, B, C = y

    if t < 0.2 * total_time:
        input_signal = .3
    elif t < 0.4 * total_time:
        input_signal = .4
    elif t < 0.6 * total_time:
        input_signal = .3
    elif t < 0.8 * total_time:
        input_signal = .4
    else:
        input_signal = .3
    dA_dt =  input_signal * kia * ((1 - A) / ((1 - A) + Kia)) - Fa * kfaa * (A / (A + Kfaa))
    dB_dt = C * kcb * ((1 - B) / ((1 - B) + Kcb)) - Fb * kfbb * (B / (B + Kfbb))
    dC_dt = A * kac * ((1 - C) / ((1 - C) + Kac)) - B * kbc * (C / (C + Kbc))
    
    return [dA_dt, dB_dt, dC_dt]

def integrate_system(y0, time_steps, params):
    global total_time
    total_time = time_steps[-1]
    result = odeint(
        lambda y, t: coupled_system(y, t, 
                                     params['I'], params['kia'], params['Kia'], 
                                     params['Fa'], params['kfaa'], params['Kfaa'], 
                                     params['kcb'], params['Kcb'], 
                                     params['Fb'], params['kfbb'], params['Kfbb'], 
                                     params['kac'], params['Kac'], 
                                     params['kbc'], params['Kbc']),
        y0, time_steps
    )
    return result

def parameter_sweep(y0, time_steps, param_sets):
    results = []
    for params in param_sets:
        result = integrate_system(y0, time_steps, params)
        results.append(result)
    return results

def main():
    y0 = [.0665, .63095, .0998]
    time_steps = np.linspace(0, 500, 100000)

    param_sets = [
            {'I': 1, 'kia': 0.1, 'Kia': 0.001, 'Fa': 0.1, 'kfaa': 1, 'Kfaa': 0.1, 'kcb': 1, 'Kcb': 0.0001, 'Fb': 1, 'kfbb': 0.1, 'Kfbb': 0.001, 'kac': 10, 'Kac': 1, 'kbc': 1, 'Kbc': 0.1}
            ]
    
    all_results = parameter_sweep(y0, time_steps, param_sets)
    
    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:cyan']
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(all_results):
        plt.plot(time_steps, result[:, 2], label=f'Sweep Param Set {i+1}', color=colors[i % len(colors)])
    
    plt.axvline(x=0.2*total_time, color='limegreen', label='Signal On', linestyle='--')
    plt.axvline(x=0.4*total_time, color='red', label='Signal Off', linestyle='--')
    plt.axvline(x=0.6*total_time, color='limegreen', linestyle='--')
    plt.axvline(x=0.8*total_time, color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration of C')
    plt.title('Concentration of C over Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/negative_basic_model_Emily.pdf')

if __name__ == "__main__":
    main()