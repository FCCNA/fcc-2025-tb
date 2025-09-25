# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import yaml
from functions import x_times, create_2d_histogram
import random
import argparse
# %%
parser = argparse.ArgumentParser(description='Process run data for analysis')
parser.add_argument('--run', type=str, help='Run number to process')
args = parser.parse_args()

run = args.run
# %%

df = pd.read_pickle(f'pickles/run{run}_wf.pkl')
yaml = yaml.safe_load(open(f'yamls/run{run}_info.yaml'))
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
print('Le variabili sono:')
for col in df_chambers.columns.levels[0]:
    print(col, list(df_chambers[col].keys()))
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


fig, axes = plt.subplots(1, 2, figsize=(24, 10))

fig.suptitle(f'Run {run} - Waveforms (Volt)', fontsize=16)
chambers = ['Cherenkov', 'Scintillator']
plot_map = {'cmin': 1e-3, 'cmap': 'plasma'}
for i, chamber in enumerate(chambers):
    wf = df_chambers[chamber]['WF']
    all_times = np.concatenate([np.arange(1, len(wf[j]) + 1) for j in wf.keys()])
    all_times = times[all_times - 1]  # Convert to actual time values
    all_wf = np.concatenate([wf[j] for j in wf.keys()])
    axes[i].hist2d(all_times, all_wf, bins=300, **plot_map)
    cbar = plt.colorbar(axes[i].collections[0], ax=axes[i])
    axes[i].set_xlabel('Times')
    axes[i].set_ylabel('WF')
    axes[i].set_title(f'{chamber}')

plt.tight_layout()
plt.savefig(f'check_plot/{run}/Waveforms_ChSc_2Dhist.png')

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle(f'Run {run} - Waveforms (ADC Units)', fontsize=16)

chambers = ['Cherenkov', 'Scintillator']
plot_map = {'cmin': 1e-3, 'cmap': 'plasma'}
for i, chamber in enumerate(chambers):
    wf = df_chambers[chamber]['WF']/yaml[chamber]['adctovolts'] + df_chambers[chamber]['pedestal']
    all_times = np.concatenate([np.arange(1, len(wf[j]) + 1) for j in wf.keys()])
    all_times = times[all_times - 1]  # Convert to actual time values
    all_wf = np.concatenate([wf[j] for j in wf.keys()])
    axes[i].hist2d(all_times, all_wf, bins=300, **plot_map)
    cbar = plt.colorbar(axes[i].collections[0], ax=axes[i])
    axes[i].set_xlabel('Times')
    axes[i].set_ylabel('WF')
    axes[i].set_title(f'{chamber}')

plt.tight_layout()
plt.savefig(f'check_plot/{run}/Waveforms_ChSc_ADC_2Dhist.png')
# %%
