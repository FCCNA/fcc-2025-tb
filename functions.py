import numpy as np
from tqdm import tqdm 
import midas.file_reader
from datetime import datetime
import os
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='palatino')
import pandas as pd
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


def convert(wf, adtovolt = 1, dcoffset = 0):
    return wf * adtovolt + dcoffset

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


def read_waveform(run_index, path, om, fc, json_data, save_waveforms):
    run = f'run{run_index:05d}'
    mfile = midas.file_reader.MidasFile(path + run + '.mid.gz', use_numpy=True)

    try:
        # Try to find the special midas event that contains an ODB dump.
        odb = mfile.get_bor_odb_dump()
        odb_dict = odb.data;

        # The full ODB is stored as a nested dict withing the `odb.data` member.
        run_number = odb_dict["Runinfo"]["Run number"]
        start_date = datetime.strptime(odb_dict["Runinfo"]["Start time"], '%a %b %d %H:%M:%S %Y')
        stop_date = datetime.strptime(odb_dict["Runinfo"]["Stop time"], '%a %b %d %H:%M:%S %Y')
        run_description = odb_dict["Runinfo"]["Run description"]
        SARA = odb_dict['Equipment']['FeLibFrontend']['Status']['Digitizer']['adc_samplrate']
        bit = 2**odb_dict['Equipment']['FeLibFrontend']['Status']['Digitizer']['adc_nbit']
        print("We are looking at a file from run number %s" % run_number)
    except RuntimeError:
        # No ODB dump found (mlogger was probably configured to not dump
        # the ODB at the start of each subrun).
        print("No begin-of-run ODB dump found")


    mfile.jump_to_start()

    print("Reading ODB")
    # Extract channel configuration from ODB
    info_channel = {key: {} for key in ["gainfactor", "adctovolts", "dcoffset"]}
    
    for channel_id in json_data["Channels"]:
        for key in info_channel:
            info_channel[key][channel_id] = odb_dict["Equipment"]["FeLibFrontend"]["Settings"][f"Channel{channel_id}"][key]
    
    total_events = sum(1 for _ in mfile if not _.header.is_midas_internal_event() and _.header.event_id == 1)
    mfile.jump_to_start()

    bit = 2**16
    
    new_data_list = []
    ch_list = []
    if fc: 
        ch_list = ["Scintillator", "Cherenkov"]
    else:
        for channel_id in json_data["Channels"]: 
            ch_list.append(json_data["Channels"][channel_id]["name"])

    progress_bar = tqdm(total=total_events, colour='green', desc="Processing events")
    progress_bar.set_postfix(evts=total_events)
    data_for_df = []
    if om:
        hist_wf = {}
        for channel_id in json_data["Channels"]:
            name = json_data["Channels"][channel_id]["name"]
            hist_wf[name] = []
    for event in mfile:
        if event.header.is_midas_internal_event():
            continue
        if event.header.event_id != 1:
            continue
        
        progress_bar.update(1)
        event_data = {}
        
        # Add generic info
        event_data[('Generic_Info', 'Run_Number')] = run_number
        event_data[('Generic_Info', 'Event_Number')] = event.header.serial_number
        event_data[('Generic_Info', 'Start_Date')] = start_date
        event_data[('Generic_Info', 'Stop_Date')] = stop_date
        event_data[('Generic_Info', 'Run_Description')] = run_description

        for bank_name, bank in event.banks.items():
            # Skip banks that don't start with 'W'
            if not bank_name.startswith('W'):
                continue
            
            # Skip event banks
            if bank_name[1:4] == '0EV':
                continue
            
            bank_index = int(bank_name[1:4])

            if str(bank_index) not in json_data["Channels"]:
                continue
            
            name = json_data["Channels"][str(bank_index)]["name"]
            if fc and name not in ch_list:
                continue
            wf = convert(bank.data, info_channel["adctovolts"][str(bank_index)], info_channel["dcoffset"][str(bank_index)])
            wf_mediata = media_mobile(wf, window_size=20)
            #print(wf)
            if om:
                hist_wf[name].extend((i/SARA, sample) for i, sample in enumerate(wf))
                
            ampl = np.max(np.abs(wf))
            ampl_mediata = np.max(np.abs(wf_mediata))
            for key in info_channel:
                event_data[(name, key)] = info_channel[key][str(bank_index)]

            event_data[(name, 'Amplitude')] = ampl
            event_data[(name, 'Amplitude_Mediata')] = ampl_mediata
            event_data[(name, 'TMax')] = np.argmax(np.abs(wf))
            event_data[(name, 'Charge')] = np.sum(wf)
            if save_waveforms:
                event_data[(name, 'WF')] = wf

            threshold = 0.1 * ampl
            t0_indices = np.where(np.abs(wf) >= threshold)[0]
            if len(t0_indices) > 0:
                event_data[(name, 't0')] = t0_indices[0]
            else:
                event_data[(name, 't0')] = 0
            event_data[(name, 'Length')] = len(wf)

        event_data[('Generic_Info', 'Times')] = 1.e9 * np.arange(0, 1 / SARA * len(wf), 1 / SARA)
        
        data_for_df.append(event_data)

    progress_bar.close()
    if not fc:
        os.makedirs('pickles', exist_ok=True)
        output_df = pd.DataFrame(data_for_df)
        output_df.columns = pd.MultiIndex.from_tuples(output_df.columns)
        add = '_wf' if save_waveforms else ''
        output_df.to_pickle(f'pickles/run{run_index}{add}.pkl')
        print(f"DataFrame for run {run_index} saved with {output_df.shape[0]} events.")

    if om:
        print("Saving Plots for Checks")
        os.makedirs('check_plot', exist_ok=True)

        # Create canvas for Cherenkov and Scintillator
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(f"Cherenkov and Scintillator Waveforms - Run {run_index}")
        
        for idx, detector in enumerate(['Cherenkov', 'Scintillator']):
            if len(hist_wf[detector]) > 0:
                axes[idx].set_title(f"{detector} Waveforms")
                axes[idx].set_xlabel("Time (ns)")
                x_bins = len(wf) + 1
                im = axes[idx].hist2d(*zip(*hist_wf[detector]), cmap='viridis', bins=[x_bins, 200])
                #up = json_data['Channels'].get(channel_id, {}).get('Upper_Limit', json_data["Upper_Limit"])
                #low = json_data['Channels'].get(channel_id, {}).get('Lower_Limit', json_data["Lower_Limit"])
                #plt.axhline(up, color='black', linewidth=0.8)
                #plt.axhline(low, color='black', linewidth=0.8)
                axes[idx].set_ylabel("Amplitude (mV)")
                axes[idx].set_xlabel("Time (ns)")
                plt.colorbar(im[3], ax=axes[idx], label='Counts')
        
        plt.tight_layout()
        plt.savefig(f'check_plot/cherenkov_scintillator_run{run_index}.png', dpi=300)
        print(f'Saved check_plot/cherenkov_scintillator_run{run_index}.png')
        plt.close()

        
        # Create canvas for Cherenkov and Scintillator
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Cherenkov and Scintillator Amplitudes - Run {run_index}")
        for idx, detector in enumerate(['Cherenkov', 'Scintillator']):
            axes[idx].hist(output_df[(detector, 'Amplitude')], bins=150)
            axes[idx].set_title(f"{detector} Amplitude Distribution - Run {run_index}")
            axes[idx].set_xlabel("Amplitude")
            axes[idx].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(f'check_plot/cherenkov_scintillator_Amplitude_run{run_index}.png', dpi = 300)
        print(f'Saved check_plot/cherenkov_scintillator_Amplitude_run{run_index}.png')
        plt.close()
        if not fc:
                
            # Create 3xN canvas for other channels (3 columns, variable rows)
            other_channels = [name for channel_id in json_data["Channels"] 
                    for name in [json_data["Channels"][channel_id]["name"]] 
                    if name not in ['Cherenkov', 'Scintillator']]
            
            if other_channels:
                n_channels = len(other_channels)
                fig, axes, n_rows, n_cols = create_subplot_layout(n_channels)
                
                fig.suptitle(f"Other Channels Waveforms - Run {run_index}")
                
                for idx, name in enumerate(other_channels):
                    if len(hist_wf.get(name, [])) > 0:
                        axes[idx].set_title(f"{name} Waveforms")
                        axes[idx].set_xlabel("Time (ns)")
                        x_bins = np.max([i for i, _ in hist_wf[name]]) + 1
                        im = axes[idx].hist2d(*zip(*hist_wf[name]), cmap='viridis', bins=[x_bins, 200])
                        plt.colorbar(im[3], ax=axes[idx], label='Counts')
                    else:
                        axes[idx].text(0.5, 0.5, f"No data for {name}", ha='center', va='center', transform=axes[idx].transAxes)
                
                # Hide unused subplots (only for layouts with multiple subplots)
                if n_channels > 1:
                    for idx in range(n_channels, n_rows * n_cols):
                        axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(f'check_plot/other_channels_run{run_index}.png', dpi=300)
                print(f'Saved check_plot/other_channels_run{run_index}.png')
                plt.close()
                
                # Create amplitude plots for other channels with same layout
                fig, axes, n_rows, n_cols = create_subplot_layout(n_channels)
                
                fig.suptitle(f"Other Channels Amplitudes - Run {run_index}")
                
                for idx, name in enumerate(other_channels):
                    if (name, 'Amplitude') in output_df.columns:
                        axes[idx].hist(output_df[(name, 'Amplitude')], bins=100, alpha=0.7)
                        axes[idx].set_title(f"{name} Amplitude Distribution")
                        axes[idx].set_xlabel("Amplitude")
                        axes[idx].set_ylabel("Counts")
                    else:
                        axes[idx].text(0.5, 0.5, f"No amplitude data for {name}", ha='center', va='center', transform=axes[idx].transAxes)
                
                # Hide unused subplots (only for layouts with multiple subplots)
                if n_channels > 1:
                    for idx in range(n_channels, n_rows * n_cols):
                        axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(f'check_plot/other_channels_amplitude_run{run_index}.png', dpi=300)
                print(f'Saved check_plot/other_channels_amplitude_run{run_index}.png')
                plt.close()
                
            else:
                print(f"No waveform data collected for run {run_index}")
