# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import yaml
from functions import x_times, create_2d_histogram, plot_waveforms_2d
import random
import argparse
# %%
parser = argparse.ArgumentParser(description='Process run data for analysis')
parser.add_argument('--run', type=str, help='Run number to process')
parser.add_argument('--path', type=str, help='Path to the data files', default='./')

args = parser.parse_args()

run = args.run
# %%

df = pd.read_pickle(f'{args.path}pickles/run{run}_wf.pkl')
yaml = yaml.safe_load(open(f'{args.path}yamls/run{run}_info.yaml'))
times = x_times(yaml['Pretrigger_Time'], yaml['WF_Length'], yaml['WF_Length_Time'])
# %%

valid_chambers = (
                (df[('Chamber_x_1', 'amplitude')] > 0.5) & 
                (df[('Chamber_x_2', 'amplitude')] > 0.5) &
                (df[('Chamber_y_1', 'amplitude')] > 0.5) & 
                (df[('Chamber_y_2', 'amplitude')] > 0.5)
            )
df_chambers = df[valid_chambers]
# %%

#   print('Le variabili sono:')
#   for col in df_chambers.columns.levels[0]:
#       print(col, list(df_chambers[col].keys()))
'''
Le variabili sono:
Chamber_x_1 ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'WF']
Chamber_x_2 ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'WF']
Chamber_y_1 ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'WF']
Chamber_y_2 ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'WF']
Cherenkov ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'amplitude_mediata', 'charge', 'WF']
Generic_Info ['Run_Number', 'Event_Number', 'wc_x', 'wc_y']
PMT ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'WF']
Plastico ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'WF']
Scintillator ['pedestal', 'is_satur', 'amplitude', 't0', 'tmax', 'amplitude_mediata', 'charge', 'WF']
'''
# %%
'''
fig, axes = plt.subplots(1, 1, figsize=(10, 8))
create_2d_histogram(axes, df_chambers['Generic_Info']['wc_x'], df_chambers['Generic_Info']['wc_y'], 'Run Position from Chambers', run, colorbar=True)

plt.tight_layout()
plt.show()
# %%
# Get the first waveform with amplitude > 0.7
mask = df_chambers['WF_Plastic', 'amplitude'] > 0.7
first_high_amplitude_idx = mask.idxmax() if mask.any() else None

if first_high_amplitude_idx is not None:
    waveform_data = df_chambers['WF_Plastic', 'WF'].loc[first_high_amplitude_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, waveform_data, 'b-', color='teal', linewidth=1)
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title(f'Event {first_high_amplitude_idx} - WF Plastic')
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No waveform found with amplitude > 0.7")
# %%

'''
# Usage
plot_waveforms_2d(df_chambers, times, yaml, run, use_adc=False)  # Volts
plot_waveforms_2d(df_chambers, times, yaml, run, use_adc=True)   # ADC Units
