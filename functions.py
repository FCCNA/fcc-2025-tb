import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rc('font', family='palatino')
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import variables as var
import yaml
from cycler import cycler  # Permette di impostare cicli di colori

# Ottieni i colori dalla palette Pastel1
pastel_colors = plt.get_cmap("Set2").colors

# Imposta lo stile dei colori per tutti i plot
plt.rcParams["axes.prop_cycle"] = cycler(color=pastel_colors)

def media_mobile(arr, window_size):
    if window_size < 1:
        raise ValueError("La dimensione della finestra deve essere almeno 1.")
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

def convert(wf, adtovolt = 1, piedistallo = 0):
    a = (wf - piedistallo) * adtovolt
    return a

def remove_bkg(df, nsigma = 5):
    for i in ['Scintillator', 'Cherenkov']:
        df = df[df[f'{i}_amplitude'] > nsigma * df[f'{i}_pedestal_std']]
    return df

def custom_plot_layout(title="", xlabel="", ylabel="", figsize=(16, 9), isWaveform = False, crystal = None, beam = None, angle = None, channels = None):
    
    string_top_left = f"$\it FCC\,Napoli - {crystal} -$" + f" {beam}" + f" - TB2025"
    string_top_right = f"{angle}°"
    
    if not isWaveform:
        plt.figure(figsize=figsize)
        plt.text(0.01, 1.01, string_top_left, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='left')
        if angle != None:
            plt.text(1, 1.01, string_top_right, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='right')
        plt.title(title)
    else:
        fig, ax = plt.subplots(len(channels), 1, sharex = True, figsize = (21, 9*len(channels)))
        for i, channel in enumerate(channels):
            ax[i].set_ylabel(ylabel, fontsize = 20)
            ax[i].grid(True)
            ax[i].text(0.01, 1.01, string_top_left,  transform=ax[i].transAxes, fontsize=25,  fontstyle = 'italic', verticalalignment='bottom', horizontalalignment='left')
            if angle != None:
                ax[i].text(1, 1.01, string_top_right + f' - Channel {channel}',  transform=ax[i].transAxes, fontsize=25,  fontstyle = 'italic', verticalalignment='bottom', horizontalalignment='right')
        plt.subplots_adjust(hspace=0.05)
        plt.title(title)    
    plt.xlabel(xlabel, fontsize = 25)
    plt.ylabel(ylabel, fontsize = 25)

def create_subplot_layout(n_channels):
    if n_channels == 1:
        n_cols = 1
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 6))
        axes = [axes]  # Make it a list for consistent indexing
    elif n_channels in [2, 4]:
        n_cols = 2
        n_rows = (n_channels + n_cols - 1) // n_cols  # Ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
    else:
        n_cols = 3
        n_rows = (n_channels + n_cols - 1) // n_cols  # Ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
    
    return fig, axes, n_rows, n_cols

def is_saturato(wf, wf_volt, bit = 14):
    threshold = 16381
    return (np.max(wf) >= threshold or np.min(wf) <= 2) or (np.max(np.abs(wf_volt)) >= 1.0)

def create_beam_position_plot(delta_x, delta_y, title, filename, verbose = False, figsize=(10, 8), bins=250, range_vals=(-40, 40)):

    fig, axes = plt.subplots(2, 2, figsize=figsize, 
                            gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [4, 1],
                                        'wspace': 0.02, 'hspace': 0.02})

    plot_map = {"cmin": 1e-6, "cmap": "plasma"}
    ax_main = axes[0, 0]
    h = ax_main.hist2d(delta_x, delta_y, bins=bins, range=(range_vals, range_vals), **plot_map)
    ax_main.set_xlabel("$\\Delta X$ (mm)", fontsize=12)
    ax_main.set_ylabel("$\\Delta Y$ (mm)", fontsize=12)
    ax_main.set_title(title, fontsize=14)
    ax_main.grid(True, alpha=0.3)

    ax_x = axes[1, 0]
    ax_x.hist(delta_x, bins=bins, range=range_vals, color=pastel_colors[0])
    ax_x.set_xlabel("$\\Delta X$ (mm)", fontsize=12)
    ax_x.set_ylabel("Counts", fontsize=12)
    ax_x.grid(True, alpha=0.3)
    ax_x.sharex(ax_main)

    ax_y = axes[0, 1]
    ax_y.hist(delta_y, bins=bins, range=range_vals, orientation='horizontal', color=pastel_colors[1])
    ax_y.set_xlabel("Counts", fontsize=12)
    ax_y.grid(True, alpha=0.3)
    ax_y.sharey(ax_main)

    axes[1, 1].set_visible(False)

    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_yticklabels(), visible=False)
    plt.savefig(filename, dpi=300)
    if verbose:
        print(f'Saved {filename}')
    plt.close()

def create_amplitude_histo(output_df, run_index, paths, logbook_entry, suffix=""):

    title_suffix = f" - {suffix}" if suffix else ""
    file_suffix = f"_{suffix}" if suffix else ""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Cherenkov and Scintillator Amplitudes{title_suffix} - Run {run_index}")
    bools_range = (run_index >= 869 and run_index <= 951) or (run_index >= 993) 
    ranges = (0, 0.15) if bools_range else (0, 1.2)
    for idx, detector in enumerate(['Cherenkov', 'Scintillator']):
        axes[idx].hist(output_df[f'{detector}_amplitude'], bins=200, range = ranges)
        axes[idx].set_yscale('log')
        axes[idx].set_title(f"{detector} Amplitude Distribution{title_suffix} - Run {run_index} - {logbook_entry['particle']} - {logbook_entry['crystal']} - {logbook_entry['angle']}")
        axes[idx].set_xlabel("Amplitude")
        axes[idx].set_ylabel("Counts")
        axes[idx].grid(True, alpha=0.3, which = 'both')
        #mean_amplitude = output_df[(detector, 'amplitude')].mean()
        #std_amplitude = output_df[(detector, 'amplitude')].std()
        
        #axes[idx].axvline(x=mean_amplitude, color='red', linestyle='--', alpha=0.7, 
        #                    label=f'Mean: {mean_amplitude:.3f}')
        #axes[idx].axvspan(mean_amplitude - std_amplitude, mean_amplitude + std_amplitude, 
        #                    alpha=0.2, color='red', label=f'±1σ: {std_amplitude:.3f}')
        axes[idx].text(0.95, 0.95, f'Events: {len(output_df)}', 
                        transform=axes[idx].transAxes, fontsize=10, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        #axes[idx].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{paths}/cherenkov_scintillator_Amplitude{file_suffix}.png', dpi=300)
    print(f'Saved {paths}/cherenkov_scintillator_Amplitude{file_suffix}.png')
    plt.close()

def create_2d_histogram(ax, delta_x, delta_y, title, run_index, colorbar = False):

    plot_map = {"cmin": 1e-6, "cmap": "plasma"}
    h = ax.hist2d(delta_x, delta_y, bins=250, range=((-40, 40), (-40, 40)), **plot_map)
    if colorbar:
        plt.colorbar(h[3], ax=ax, label='Events')
    ax.set_title(f"{title} - Run {run_index}", fontsize=14)
    ax.set_xlabel("$\\Delta X$ (mm)", fontsize=12)
    ax.set_ylabel("$\\Delta Y$ (mm)", fontsize=12)
    ax.grid(True, alpha=0.3)
    return h

def plot_waveforms_2d(df_chambers, times, yaml, run, cut = None, use_adc=False):

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    unit_type = "ADC Units" if use_adc else "Volt"
    fig.suptitle(f'Run {run} - Waveforms ({unit_type})', fontsize=16)
    
    chambers = ['Cherenkov', 'Scintillator']
    plot_map = {'cmin': 1e-3, 'cmap': 'plasma'}
    
    for i, chamber in enumerate(chambers):
        print(f'Plotting {chamber} waveforms, with {"ADC" if use_adc else "Volt"} scale')
        if use_adc:
            wf = df_chambers[f'{chamber}_WF']/yaml[chamber]['adctovolts'] + df_chambers[f'{chamber}_pedestal']
        else:
            wf = df_chambers[f'{chamber}_WF']
            
        all_times = np.concatenate([np.arange(1, len(wf[j]) + 1) for j in wf.keys()])
        all_times = times[all_times - 1]  # Convert to actual time values
        all_wf = np.concatenate([wf[j] for j in wf.keys()])
        
        axes[i].hist2d(all_times, all_wf, bins=300, **plot_map)
        cbar = plt.colorbar(axes[i].collections[0], ax=axes[i])
        axes[i].set_xlabel('Times')
        axes[i].set_ylabel('WF')
        axes[i].set_title(f'{chamber} - {cut}')

    plt.tight_layout()
    
    suffix = "ADC_2Dhist" if use_adc else "2Dhist"
    if cut:
        suffix+= f"_{cut}"
    plt.savefig(f'check_plot/{run}/Waveforms_ChSc_{suffix}.png')
    print(f'Saved check_plot/{run}/Waveforms_ChSc_{suffix}.png')

def create_efficiency_plot(output_df, list_of_cuts, run_index, save_path, name, logbook_entry):

    effs = []
    errs = [] 
    cumulative_mask = np.ones(len(output_df), dtype=bool)
    
    for cut_name, cut_condition in list_of_cuts.items():
        if cut_condition is not None:
            cumulative_mask &= cut_condition
        
        output_df_cut = output_df[cumulative_mask]
        eps = len(output_df_cut) / len(output_df)
        err = np.sqrt(eps * (1 - eps) / len(output_df)) if len(output_df) > 0 else 0
        effs.append(eps)
        errs.append(err)
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(effs)), effs, yerr=errs, fmt='o')
    for i, (eff, err) in enumerate(zip(effs, errs)):
        plt.annotate(f'{eff:.3f}', (i, eff), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xticks(range(len(effs)), list(list_of_cuts.keys()))
    plt.ylim(0, 1.1)
    plt.xlabel('Selection Stage')
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency Plot for Run {run_index} - {logbook_entry["particle"]} - {logbook_entry["crystal"]} - {logbook_entry["angle"]}')
    plt.grid(True)
    plt.savefig(f'{save_path}/efficiency_{name}.png', dpi=300)
    print(f'Saved {save_path}/efficiency_{name}.png')
    plt.close()

def find_t0(wf, threshold, times):
    t0_indices = np.where(np.abs(wf) >= threshold)[0]
    
    if len(t0_indices) > 0:
        idx = t0_indices[0]
        if idx > 0:
            y1, y2 = np.abs(wf[idx-1]), np.abs(wf[idx])
            x1, x2 = times[idx-1], times[idx]
            t0_interp = x1 + (threshold - y1) / (y2 - y1) * (x2 - x1)
            return t0_interp
        else:
            return times[idx]
    else:
        return 0
    
def apply_data_cuts(output_df, cut, run_number):
    plastico_cut = 0.01 if (run_number >= 879 and run_number <= 896) else 0.1
    if cut == 'chambers':
        filter = (
                (output_df['Chamber_x_1_amplitude'] > 0.5) &
                (output_df['Chamber_x_2_amplitude'] > 0.5) &
                (output_df['Chamber_y_1_amplitude'] > 0.5) & 
                (output_df['Chamber_y_2_amplitude'] > 0.5)
            )

    elif cut == 'plastico':
        filter = (
            (output_df['Chamber_x_1_amplitude'] > 0.5) &
            (output_df['Chamber_x_2_amplitude'] > 0.5) &
            (output_df['Chamber_y_1_amplitude'] > 0.5) & 
            (output_df['Chamber_y_2_amplitude'] > 0.5) &
            (output_df['Plastico_amplitude'] > plastico_cut)
        )
    elif cut == 'crystal':
        filter = (
            (output_df['Chamber_x_1_amplitude'] > 0.5) &
            (output_df['Chamber_x_2_amplitude'] > 0.5) &
            (output_df['Chamber_y_1_amplitude'] > 0.5) & 
            (output_df['Chamber_y_2_amplitude'] > 0.5) &
            (output_df['Plastico_amplitude'] > plastico_cut) &
            (output_df['wc_x'].between(-5, 5)) &
            (output_df['wc_y'].between(7, 12))
        )
    else:
        print(f'No valid cut selected - {cut}')
        return output_df
    
    results = output_df[filter]
    return results

def x_times(pretrigger_time, length_time, length):
    return np.arange(0-pretrigger_time, length_time - pretrigger_time, length_time / length)