import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def coupled_system(y, t, I, kia, Kia, Fa, kfaa, Kfaa, kcb, Kcb, Fb, kfbb, Kfbb, kac, Kac, kbc, Kbc, kab, Kab):
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
        
    dA_dt = input_signal * kia * ((1 - A) / ((1 - A) + Kia)) - Fa * kfaa * (A / (A + Kfaa))
    dB_dt = C * kcb * ((1 - B) / ((1 - B) + Kcb)) - Fb * kfbb * (B / (B + Kfbb)) - A * kab * (B / (B + Kab))
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
                                     params['kbc'], params['Kbc'], params['kab'], params['Kab']),
        y0, time_steps
    )
    return result

def parameter_sweep(y0, time_steps, param_sets):
    results = []
    for params in param_sets:
        result = integrate_system(y0, time_steps, params)
        results.append(result)
    return results

def find_first_zero_crossings(dC_dt, time_steps, start_time):
    start_index = np.searchsorted(time_steps, start_time)
    for i in range(start_index + 1, len(dC_dt)):
        if dC_dt[i-1] * dC_dt[i] < 0:
            return i
    return None

def main():
    y0 = [.1, .1, .5]
    time_steps = np.linspace(0, 1000, 1000)

    param_sets = [
        {'I': 1, 'kia': 5, 'Kia': 20, 'Fa': 0.5, 'kfaa': 1, 'Kfaa': 1, 'kcb': 0.1, 'Kcb': 0.01, 'Fb': 0.5, 'kfbb': 0.1, 'Kfbb': 0.01, 'kac': 10, 'Kac': 1, 'kbc': 5, 'Kbc': 0.5, 'kab': 0.01, 'Kab': 1}
    ]
    
    all_results = parameter_sweep(y0, time_steps, param_sets)
    
    dC_dt_all = []
    total_time = time_steps[-1]
    first_time_point = 0.3 * total_time
    second_time_point = 0.5 * total_time

    percents = []

    for params, result in zip(param_sets, all_results):
        A_values = result[:, 0]
        B_values = result[:, 1]
        C_values = result[:, 2]
        
        dC_dt = np.array([
            coupled_system([A, B, C], t, **params)[2] 
            for A, B, C, t in zip(A_values, B_values, C_values, time_steps)
        ])
        dC_dt_all.append(dC_dt)
        
        index_0_3 = find_first_zero_crossings(dC_dt, time_steps, first_time_point)
        index_0_5 = find_first_zero_crossings(dC_dt, time_steps, second_time_point)
        if index_0_3 is not None and index_0_5 is not None:
            C_value_0_3 = result[300, 2]
            C_value_0_5 = result[500, 2]
            print(C_value_0_3, C_value_0_5)
            percent_difference = (abs(C_value_0_3 - C_value_0_5) / ((C_value_0_3 + C_value_0_5) / 2)) * 100
            percents.append(percent_difference)
            print(f'Percent Difference for Parameter Set {params}: {percent_difference:.2f}%')

    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:cyan']
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(all_results):
        plt.plot(time_steps, result[:, 2], label=f'Sweep Param Set {i+1},  {str(round(percents[i], 2))}%', color=colors[i % len(colors)])
    
    plt.axvline(x=0.2*total_time, color='limegreen', label='0.2 * total_time', linestyle='--')
    plt.axvline(x=0.4*total_time, color='red', label='0.4 * total_time', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration of C')
    plt.title('Concentration of C over Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.grid()
    plt.tight_layout()

    plt.savefig('results/negative_A_regulates_B_model.pdf')

if __name__ == "__main__":
    main()
