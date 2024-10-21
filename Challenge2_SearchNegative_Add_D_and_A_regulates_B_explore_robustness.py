import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import product

def coupled_system(y, t, I, kia, Kia, Fa, kfaa, Kfaa, kcb, Kcb, Fb, kfbb, Kfbb, kac, Kac, kbc, Kbc, kad, Kad, Fd, kfdd, Kfdd, kcd, Kcd, kdc, Kdc, kab, Kab):
    A, B, C, D = y
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
        
    dA_dt = input_signal * kia * ((1 - A) / ((1 - A) + Kia)) - Fa * kfaa * (A / (A + Kfaa))
    dB_dt = C * kcb * ((1 - B) / ((1 - B) + Kcb)) - Fb * kfbb * (B / (B + Kfbb)) - A * kab * (B / (B + Kab))
    dC_dt = A * kac * ((1 - C) / ((1 - C) + Kac)) - B * kbc * (C / (C + Kbc)) + D * kdc * ((1 - C) / ((1 - C) + Kdc))
    dD_dt = A * kad * ((1 - D) / ((1 - D) + Kad)) - Fd * kfdd * (D / (D + Kfdd)) - C * kcd * (D / (D + Kcd))
    
    return [dA_dt, dB_dt, dC_dt, dD_dt]

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
                                     params['kbc'], params['Kbc'], 
                                     params['kad'], params['Kad'], 
                                     params['Fd'], params['kfdd'], params['Kfdd'], 
                                     params['kcd'], params['Kcd'], 
                                     params['kdc'], params['Kdc'],
                                     params['kab'], params['Kab']),
        y0, time_steps
    )
    return result

def calculate_percent_difference(C_value_0_3, C_value_0_5):
    return (abs(C_value_0_3 - C_value_0_5) / ((C_value_0_3 + C_value_0_5) / 2)) * 100

def plot_all_results(time_steps, all_results, param_sets, param_name, param_index, percents):
    plt.figure(figsize=(10, 6))
    colors = ['fuchsia','darkorange','gold','cyan', 'deepskyblue', 'royalblue', 'blueviolet', 'mediumvioletred']
    for i, (result, params) in enumerate(zip(all_results, param_sets)):
        plt.plot(time_steps, result[:, 2], color=colors[i % len(colors)],
                 label=f'{param_name} = {params[param_name]}, Diff = {percents[i]:.2f}%')
    plt.axvline(x=0.2*total_time, color='limegreen', label='Signal On', linestyle='--')
    plt.axvline(x=0.4*total_time, color='red', label='Signal Off', linestyle='--')
    plt.axvline(x=0.6*total_time, color='limegreen', linestyle='--')
    plt.axvline(x=0.8*total_time, color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration of C')
    plt.title('Concentration of C over Time')
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    fname = f'results/NFBLB_add_D_and_A_regulates_B_explore_{param_name}_{param_index}_sweep.pdf'
    print(f'Saving plot: {fname}')
    plt.savefig(fname)
    plt.close()
    
def parameter_sweep(y0, time_steps, param_values, param_name, param_index):
    all_results = []
    param_sets = []
    percents = []

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
            'kad': params[15],
            'Kad': params[16],
            'Fd': params[17],
            'kfdd': params[18],
            'Kfdd': params[19],
            'kcd': params[20],
            'Kcd': params[21],
            'kdc': params[22],
            'Kdc': params[23],
            'kab': params[24],
            'Kab': params[25]
        }
        result = integrate_system(y0, time_steps, param_dict)
        all_results.append(result)
        param_sets.append(param_dict)
        
        C_value_0_3 = result[int(0.3 * len(time_steps)), 2]
        C_value_0_5 = result[int(0.5 * len(time_steps)), 2]
        percent_difference = calculate_percent_difference(C_value_0_3, C_value_0_5)
        percents.append(percent_difference)

    plot_all_results(time_steps, all_results, param_sets, param_name, param_index, percents)
    
def main():
    y0 = [0.1, 0.1, 0.5, 0.1]  # Initial values for A, B, C, D
    time_steps = np.linspace(0, 1000, 1000)

    param_sets = [{'I': 1, 'kia': 5, 'Kia': 20, 'Fa': 0.5, 'kfaa': 1, 'Kfaa': 1, 
         'kcb': 0.1, 'Kcb': 0.01, 'Fb': 0.5, 'kfbb': 0.1, 'Kfbb': 0.01, 
         'kac': 10, 'Kac': 1, 'kbc': 5, 'Kbc': 0.5, 
         'kad': 1, 'Kad': 1, 'Fd': 0.5, 
         'kfdd': 0.1, 'Kfdd': 0.01, 'kcd': 0.1, 'Kcd': 0.01, 
         'kdc': 0.1, 'Kdc': 0.01, 'kab': 0.01, 'Kab':0.1}]
    
    base_params = {
        'I': [1, 1],  # I values
        'kia': [5],  # kia values
        'Kia': [20],  # Kia values
        'Fa': [0.1, 0.5, 1, 2, 5, 10, 20],  # Fa values
        'kfaa': [0.1, 0.5, 1, 2, 5, 10, 20, 50],  # kfaa values
        'Kfaa': [1],  # Kfaa values
        'kcb': [0.1],  # kcb values
        'Kcb': [0.01],  # Kcb values
        'Fb': [0.5],  # Fb values
        'kfbb': [0.1],  # kfbb values
        'Kfbb': [0.01],  # Kfbb values
        'kac': [1, 2, 5, 10, 20, 50, 100],  # kac values
        'Kac': [1],  # Kac values
        'kbc': [5, 10, 20, 50, 100, 200],  # kbc values
        'Kbc': [0.01, 0.05, 0.1, 0.5, 1],  # Kbc values
        'kad': [1],
        'Kad': [1],
        'Fd': [0.5],
        'kfdd': [0.1],
        'Kfdd': [0.01],
        'kcd': [0.1],
        'Kcd': [0.01],
        'kdc': [0.1],
        'Kdc': [0.01],
        'kab': [0.01],
        'Kab': [0.1]
    }

    params_list = ['kia', 'Kia', 'Fa', 'kfaa', 'Kfaa', 'kcb', 'Kcb', 'Fb', 'kfbb', 'Kfbb', 
                   'kac', 'Kac', 'kbc', 'Kbc', 'kad', 'Kad', 'Fd', 'kfdd', 'Kfdd', 'kcd',
                   'Kcd', 'kdc', 'Kdc', 'kab', 'Kab']

    for param_name in params_list:
        print(f"Processing parameter: {param_name}")
        param_values = []
        for key in base_params.keys():
            if key == param_name:
                param_values.append(base_params[key])
            else:
                param_values.append([param_sets[0][key]])
        index = params_list.index(param_name) + 1
        parameter_sweep(y0, time_steps, param_values, param_name, str(index))

if __name__ == "__main__":
    main()
