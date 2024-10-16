import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import product

def coupled_system(y, t, I, kia, Kia, Fa, kfaa, Kfaa, kcb, Kcb, Fb, kfbb, Kfbb, kac, Kac, kbc, Kbc):
    A, B, C = y
    if t < 0.25 * total_time:
        input_signal = 0.0
    elif t < 0.5 * total_time:
        input_signal = 1.0
    elif t < 0.75 * total_time:
        input_signal = 0.0
    else:
        input_signal = 1.0
    dA_dt = input_signal * kia * ((1 - A) / ((1 - A) + Kia)) - Fa * kfaa * (A / (A + Kfaa))
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

def plot_all_results(time_steps, all_results, param_sets, param_name, param_index):
    plt.figure(figsize=(10, 6))
    colors = ['cyan', 'deepskyblue', 'royalblue', 'blueviolet']
    for i, (result, params) in enumerate(zip(all_results, param_sets)):
        plt.plot(time_steps, result[:, 2], color=colors[i % len(colors)],
                 label=param_name + ' = ' + str(params[param_name]))
    plt.axvline(x=1000/4, color='tab:green', label='Signal On', linestyle='--')
    plt.axvline(x=1000/2, color='tab:red', label='Signal Off', linestyle='--')
    plt.axvline(x=3*1000/4, color='tab:green', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration of C')
    plt.title('Concentration of C over Time')
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    fname = f'results/{param_name}_{param_index}_sweep.pdf'
    print(f'Saving plot: {fname}')
    plt.savefig(fname)
    plt.close()
    
def parameter_sweep(y0, time_steps, param_values, param_name, param_index):
    all_results = []
    param_sets = []
    for params in product(*param_values):
        param_dict = {
            'I': params[0],
            'kia': params[1],
            'Kia': params[2],
            'Fa': params[3],
            'kfaa': params[4],
            'Kfaa': params[5],
            'kcb': params[6],
            'Kcb': params[7],
            'Fb': params[8],
            'kfbb': params[9],
            'Kfbb': params[10],
            'kac': params[11],
            'Kac': params[12],
            'kbc': params[13],
            'Kbc': params[14],
        }
        result = integrate_system(y0, time_steps, param_dict)
        all_results.append(result)
        param_sets.append(param_dict)
    plot_all_results(time_steps, all_results, param_sets, param_name, param_index)
    
def main():
    y0 = [.1, .1, .1]
    time_steps = np.linspace(0, 1000, 1000)

    base_params = {
        'I': [0],  # I values
        'kia': [0.01, 0.1, 1, 10],  # kia values
        'Kia': [0.01, 0.1, 1, 10],  # Kia values
        'Fa': [0.01, 0.1, 1, 10],  # Fa values
        'kfaa': [0.01, 0.1, 1, 10],  # kfaa values
        'Kfaa': [0.01, 0.1, 1, 10],  # Kfaa values
        'kcb': [0.01, 0.1, 1, 10],  # kcb values
        'Kcb': [0.01, 0.1, 1, 10],  # Kcb values
        'Fb': [0.01, 0.1, 1, 10],  # Fb values
        'kfbb': [0.01, 0.1, 1, 10],  # kfbb values
        'Kfbb': [0.01, 0.1, 1, 10],  # Kfbb values
        'kac': [0.01, 0.1, 1, 10],  # kac values
        'Kac': [0.01, 0.1, 1, 10],  # Kac values
        'kbc': [0.01, 0.1, 1, 10],  # kbc values
        'Kbc': [0.01, 0.1, 1, 10],  # Kbc values
    }

    params_list = ['kia', 'Kia', 'Fa', 'kfaa', 'Kfaa', 'kcb', 'Kcb', 'Fb', 'kfbb', 'Kfbb', 'kac', 'Kac', 'kbc', 'Kbc']
    
    for param_name in params_list:
        print(f"Processing parameter: {param_name}")
        param_values = []
        for key in base_params.keys():
            if key == param_name:
                param_values.append(base_params[key])
            else:
                param_values.append([base_params[key][0]])
        index = params_list.index(param_name) + 1
        parameter_sweep(y0, time_steps, param_values, param_name, str(index))

if __name__ == "__main__":
    main()
