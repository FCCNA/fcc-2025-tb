# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import yaml
from functions import x_times, create_2d_histogram, plot_waveforms_2d, apply_data_cuts
import random
import argparse
# %%
parser = argparse.ArgumentParser(description='Process run data for analysis')
parser.add_argument('--run', type=int, help='Run number to process')
parser.add_argument('--path', type=str, help='Path to the data files', default='./')

args = parser.parse_args()

run = args.run
# %%

col_list = ['Cherenkov_WF', 'Scintillator_WF', 
            'Cherenkov_pedestal', 'Scintillator_pedestal', 
            'Chamber_x_1_amplitude', 'Chamber_x_2_amplitude', 'Chamber_y_1_amplitude', 'Chamber_y_2_amplitude', 'Plastico_amplitude', 
            'wc_x', 'wc_y']
df = pd.read_parquet(f'{args.path}parquet/run{run}_wf.parq', columns=col_list, engine='pyarrow')
yaml_data = yaml.safe_load(open(f'{args.path}yamls/run{run}_info.yaml'))
times = x_times(yaml_data['Pretrigger_Time'], yaml_data['WF_Length'], yaml_data['WF_Length_Time'])

 

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

'''
cuts = ['chambers', 'plastico', 'crystal']
for cut in cuts:
    df_temp = apply_data_cuts(df, cut, run)
    for adc in [True, False]:
        plot_waveforms_2d(df_temp, times, yaml_data, run, use_adc=adc, cut=cut)
