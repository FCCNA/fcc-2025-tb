import numpy as np
from tqdm import tqdm 
import midas.file_reader
from datetime import datetime
import os
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='palatino')
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
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

def create_beam_position_plot(delta_x, delta_y, title, filename, figsize=(10, 8), bins=150, range_vals=(-20, 20)):

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
    print(f'Saved {filename}')
    plt.close()

def create_2d_histogram(ax, delta_x, delta_y, title, run_index, colorbar = False):

    plot_map = {"cmin": 1e-6, "cmap": "plasma"}
    h = ax.hist2d(delta_x, delta_y, bins=150, range=((-20, 20), (-20, 20)), **plot_map)
    if colorbar:
        plt.colorbar(h[3], ax=ax, label='Events')
    ax.set_title(f"{title} - Run {run_index}", fontsize=14)
    ax.set_xlabel("$\\Delta X$ (mm)", fontsize=12)
    ax.set_ylabel("$\\Delta Y$ (mm)", fontsize=12)
    ax.grid(True, alpha=0.3)
    return h

def create_efficiency_plot(output_df, list_of_cuts, run_index, save_path, name):

    effs = []
    errs = [] 
    cumulative_mask = np.ones(len(output_df), dtype=bool)
    
    for cut_name, cut_condition in list_of_cuts.items():
        if cut_condition is not None:
            cumulative_mask &= cut_condition
        
        output_df_cut = output_df[cumulative_mask]
        eps = output_df_cut.shape[0] / output_df.shape[0]
        err = np.sqrt(eps * (1 - eps) / output_df.shape[0])
        effs.append(eps)
        errs.append(err)
    
    plt.errorbar(range(len(effs)), effs, yerr=errs, fmt='o')
    for i, (eff, err) in enumerate(zip(effs, errs)):
        plt.annotate(f'{eff:.3f}', (i, eff), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xticks(range(len(effs)), list(list_of_cuts.keys()))
    plt.ylim(0, 1.1)
    plt.xlabel('Selection Stage')
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency Plot for Run {run_index}')
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

def x_times(pretrigger_time, length_time, length):
    return np.arange(0-pretrigger_time, length_time - pretrigger_time, length_time / length)
    
def read_waveform(run_index, path, om, fc, json_data, save_waveforms, avoid):
    run = f'run{run_index:05d}'
    mfile = midas.file_reader.MidasFile(path + run + '.mid.gz', use_numpy=True)
    paths = f'check_plot/{run_index}'
    os.makedirs(f'check_plot/{run_index}', exist_ok=True)

    if not avoid:

        try:
            # Try to find the special midas event that contains an ODB dump.
            odb = mfile.get_bor_odb_dump()
            odb_dict = odb.data;

            # The full ODB is stored as a nested dict withing the `odb.data` member.
            run_number = odb_dict["Runinfo"]["Run number"]
            start_date = datetime.strptime(odb_dict["Runinfo"]["Start time"], '%a %b %d %H:%M:%S %Y')
            stop_date = datetime.strptime(odb_dict["Runinfo"]["Stop time"], '%a %b %d %H:%M:%S %Y')
            run_description = odb_dict["Runinfo"]["Run description"]
            #SARA = odb_dict['Equipment']['FeLibFrontend']['Status']['Digitizer']['adc_samplrate']
            bit = 2**odb_dict['Equipment']['FeLibFrontend']['Status']['Digitizer']['adc_nbit']
            length = int(odb_dict['Equipment']['FeLibFrontend']['Settings']['Digitizer']['recordlengths'], 16)
            length_time = int(odb_dict['Equipment']['FeLibFrontend']['Settings']['Digitizer']['recordlengtht'], 16)
            pretrigger_time = int(odb_dict['Equipment']['FeLibFrontend']['Settings']['Digitizer']['pretriggert'], 16)
            pretrigger_sample = int(odb_dict['Equipment']['FeLibFrontend']['Settings']['Digitizer']['pretriggers'], 16)
            print("We are looking at a file from run number %s" % run_number)
        except RuntimeError:
            # No ODB dump found (mlogger was probably configured to not dump
            # the ODB at the start of each subrun).
            print("No begin-of-run ODB dump found")
        yaml_output = {}
        yaml_output['Run_Number'] = run_number
        yaml_output['Start_Date'] = start_date
        yaml_output['Stop_Date'] = stop_date
        yaml_output['Json'] = json_data['Description']
        yaml_output['Run_Description'] = run_description
        yaml_output['Pretrigger_Time'] = pretrigger_time
        yaml_output['Pretrigger_Sample'] = pretrigger_sample
        yaml_output['WF_Length'] = length
        yaml_output['WF_Length_Time'] = length_time
        times = x_times(pretrigger_time, length_time, length)

        mfile.jump_to_start()

        print("Reading ODB")
        # Extract channel configuration from ODB
        info_channels = ["gainfactor", "adctovolts", "dcoffset"]
        
        for channel_id in json_data["Channels"]:
            name = json_data["Channels"][channel_id]["name"]
            yaml_output[name] = {}
            for key in info_channels:
                yaml_output[name][key] = odb_dict["Equipment"]["FeLibFrontend"]["Settings"][f"Channel{channel_id}"][key]
        
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
        
        satur = {}
        for channel_id in json_data["Channels"]:
            name = json_data["Channels"][channel_id]["name"]
            satur[name] = 0

        for event in mfile:
            if event.header.is_midas_internal_event():
                continue
            if event.header.event_id != 1:
                continue
            
            progress_bar.update(1)
            event_data = {}
            event_data[('Generic_Info', 'Run_Number')] = run_number
            event_data[('Generic_Info', 'Event_Number')] = event.header.serial_number

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

                piedistallo = bank.data[:pretrigger_sample-10].mean()
                event_data[(name, 'pedestal')] = piedistallo
                wf = convert(bank.data, yaml_output[name]["adctovolts"], piedistallo)

                if is_saturato(bank.data, wf, bit = bit):
                    satur[name] += 1
                    event_data[(name, 'is_satur')] = True
                else:
                    event_data[(name, 'is_satur')] = False
                

                ampl = np.max(np.abs(wf))
                event_data[(name, 'amplitude')] = ampl
                t0 = find_t0(wf, ampl*0.1, times)
                event_data[(name, 't0')] = t0
                event_data[(name, 'tmax')] = times[np.argmax(np.abs(wf))]
                

                if name in ["Scintillator", "Cherenkov"]:
                    wf_mediata = media_mobile(wf, window_size=20)
                    ampl_mediata = np.max(np.abs(wf_mediata))
                    event_data[(name, 'amplitude_mediata')] = ampl_mediata
                    event_data[(name, 'charge')] = np.sum(wf)


                if save_waveforms:
                    event_data[(name, 'WF')] = wf
            
            data_for_df.append(event_data)

        progress_bar.close()
    
        output_df = pd.DataFrame(data_for_df)
        output_df.columns = pd.MultiIndex.from_tuples(output_df.columns)
    else:
        output_df = pd.read_pickle(f'pickles/run{run_index}_wf.pkl')
        yaml_output = {}
        with open(f'yamls/run{run_index}_info.yaml', 'r') as yaml_file:
            yaml_output = yaml.safe_load(yaml_file)
        satur = {}
        for channel_id in json_data["Channels"]:
            name = json_data["Channels"][channel_id]["name"]
            satur[name] = output_df[(name, 'is_satur')].sum()
        print(f"Loaded DataFrame for run {run_index} with {output_df.shape[0]} events.")
        print(f"Saturated events for run {run_index}: {satur}")
        paths = f'check_plot/{run_index}'
        os.makedirs(f'check_plot/{run_index}', exist_ok=True)
    conversion = 0.18
    output_df['Generic_Info', 'wc_x'] = conversion * (output_df[('Chamber_x_1', 't0')] - output_df[('Chamber_x_2', 't0')])
    output_df['Generic_Info', 'wc_y'] = conversion * (output_df[('Chamber_y_1', 't0')] - output_df[('Chamber_y_2', 't0')])


    if not fc:
        os.makedirs('pickles', exist_ok=True)
        # Write yaml_output to file
        yaml_filename = f'yamls/run{run_index}_info.yaml'
        os.makedirs('yamls', exist_ok=True)
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(yaml_output, yaml_file)
        print(f"YAML info for run {run_index} saved to {yaml_filename}")
        add = '_wf' if save_waveforms else ''
        output_df.to_pickle(f'pickles/run{run_index}{add}.pkl')
        print(f"DataFrame for run {run_index} saved with {output_df.shape[0]} events.")
        print(f"Saturated events for run {run_index}: {satur}")
        print("Creating Efficiency Plots")
        if om:
            list_of_cuts_plastic = {
                'Trigger': None,
                'Plastico': (output_df[('Plastico', 'amplitude')] > 0.1),
                'Chambers': (
                    (output_df[('Chamber_x_1', 'amplitude')] > 0.5) & 
                    (output_df[('Chamber_x_2', 'amplitude')] > 0.5) &
                    (output_df[('Chamber_y_1', 'amplitude')] > 0.5) & 
                    (output_df[('Chamber_y_2', 'amplitude')] > 0.5)
                ),
            }
            create_efficiency_plot(output_df, list_of_cuts_plastic, run_index, paths, 'Scintillato')
            list_of_cuts_chambers = {
                'Trigger': None,
                'Chambers': (
                    (output_df[('Chamber_x_1', 'amplitude')] > 0.5) & 
                    (output_df[('Chamber_x_2', 'amplitude')] > 0.5) &
                    (output_df[('Chamber_y_1', 'amplitude')] > 0.5) & 
                    (output_df[('Chamber_y_2', 'amplitude')] > 0.5)
                ),
            }
            create_efficiency_plot(output_df, list_of_cuts_chambers, run_index, paths, 'chambers')

    if om:
        print("Saving Plots for Checks")

        if 'Chamber_x_1' in output_df.columns.get_level_values(0) and 'Chamber_x_2' in output_df.columns.get_level_values(0) and \
           'Chamber_y_1' in output_df.columns.get_level_values(0) and 'Chamber_y_2' in output_df.columns.get_level_values(0):
            # Create canvas for Cherenkov and Scintillator
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            # Filter out events where chamber timing is invalid
            valid_chambers = (
                (output_df[('Chamber_x_1', 'amplitude')] > 0.5) &
                (output_df[('Chamber_x_2', 'amplitude')] > 0.5) &
                (output_df[('Chamber_y_1', 'amplitude')] > 0.5) & 
                (output_df[('Chamber_y_2', 'amplitude')] > 0.5)
            )
            output_df_filtered = output_df[valid_chambers]
            output_df_filtered_plastico = output_df_filtered[output_df_filtered[('Plastico', 'amplitude')] > 0.1]
            output_df_filtered_scint = output_df_filtered[output_df_filtered[('Scintillator', 'amplitude')] > 0.05]

            create_2d_histogram(axes[0], output_df_filtered['Generic_Info']['wc_x'], output_df_filtered['Generic_Info']['wc_y'], "Beam Position from Chambers", run_index)
            create_2d_histogram(axes[1], output_df_filtered_plastico['Generic_Info']['wc_x'], output_df_filtered_plastico['Generic_Info']['wc_y'], "Beam Position from Chambers - Plastico ", run_index)
            create_2d_histogram(axes[2], output_df_filtered_scint['Generic_Info']['wc_x'], output_df_filtered_scint['Generic_Info']['wc_y'], "Beam Position from Chambers - Scintillator ", run_index)

            plt.tight_layout()
            plt.savefig(f'{paths}/chambers_beamposition_both.png', dpi=300)
            print(f'Saved {paths}/chambers_beamposition_both.png')
            plt.close()


            create_beam_position_plot(output_df_filtered['Generic_Info']['wc_x'], output_df_filtered['Generic_Info']['wc_y'], 
                                     f"Beam Position from Chambers - Run {run_index}", 
                                     f'{paths}/chambers_beamposition_grid.png')

            create_beam_position_plot(output_df_filtered_plastico['Generic_Info']['wc_x'], output_df_filtered_plastico['Generic_Info']['wc_y'], 
                                     f"Beam Position from Chambers - Plastico  - Run {run_index}", 
                                     f'{paths}/chambers_beamposition_plastico_grid.png')
        
        if 'Cherenkov' in output_df.columns.get_level_values(0) and 'Scintillator' in output_df.columns.get_level_values(0):
            fig, axes = plt.subplots(1, 2, figsize=(24, 10))
            fig.suptitle(f"Cherenkov and Scintillator Waveforms - Run {run_index}")

            #for idx, detector in enumerate(['Cherenkov', 'Scintillator']):
            #    if len(hist_wf[detector]) > 0:
            #        axes[idx].set_title(f"{detector} Waveforms")
            #        axes[idx].set_xlabel("Time (ns)")
            #        
            #        x_data, y_data = zip(*hist_wf[detector])
            #        plot_map = {
            #            'cmin': 1e-6,
            #            'cmap': 'plasma'
            #        }
            #        hist2d = axes[idx].hist2d(x_data, y_data, bins=50, **plot_map)
            #        plt.colorbar(hist2d[3], ax=axes[idx], label='Counts')
            #        
            #        axes[idx].set_ylabel("Amplitude (mV)")
            #        axes[idx].set_xlabel("Time (ns)")
        
            # plt.tight_layout()
            # plt.savefig(f'{paths}/cherenkov_scintillator.png', dpi=300)
            # print(f'Saved {paths}/cherenkov_scintillator.png')
            # plt.close()
            # Create canvas for Cherenkov and Scintillator
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"Cherenkov and Scintillator Amplitudes - Run {run_index}")
            for idx, detector in enumerate(['Cherenkov', 'Scintillator']):
                axes[idx].hist(output_df[(detector, 'amplitude')], bins=100)
                axes[idx].set_yscale('log')
                axes[idx].set_title(f"{detector} Amplitude Distribution - Run {run_index}")
                axes[idx].set_xlabel("Amplitude")
                axes[idx].set_ylabel("Counts")
            plt.tight_layout()
            plt.savefig(f'{paths}/cherenkov_scintillator_Amplitude.png', dpi = 300)
            print(f'Saved {paths}/cherenkov_scintillator_Amplitude.png')
            plt.close()
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"Cherenkov and Scintillator Amplitudes - Run {run_index}")
            for idx, detector in enumerate(['Cherenkov', 'Scintillator']):
                axes[idx].hist(output_df_filtered[(detector, 'amplitude')], bins=100)
                axes[idx].set_yscale('log')
                axes[idx].set_title(f"{detector} Amplitude Distribution - Run {run_index}")
                axes[idx].set_xlabel("Amplitude")
                axes[idx].set_ylabel("Counts")
            plt.tight_layout()
            plt.savefig(f'{paths}/cherenkov_scintillator_Amplitude_chambers.png', dpi = 300)
            print(f'Saved {paths}/cherenkov_scintillator_Amplitude_chambers.png')
            plt.close()


        if not fc:
                
            other_channels = [name for channel_id in json_data["Channels"] 
                    for name in [json_data["Channels"][channel_id]["name"]] 
                    if name not in ['Cherenkov', 'Scintillator']]
            
            if other_channels:
                n_channels = len(other_channels)

                fig, axes, n_rows, n_cols = create_subplot_layout(n_channels)
                
                fig.suptitle(f"Other Channels Amplitudes - Run {run_index}")
                
                for idx, name in enumerate(other_channels):
                    if (name, 'amplitude') in output_df.columns:
                        axes[idx].hist(output_df[(name, 'amplitude')], bins=100, alpha=0.7)
                        axes[idx].set_title(f"{name} Amplitude Distribution")
                        axes[idx].set_xlabel("Amplitude")
                        axes[idx].set_ylabel("Counts")
                    else:
                        axes[idx].text(0.5, 0.5, f"No amplitude data for {name}", ha='center', va='center', transform=axes[idx].transAxes)
                
                if n_channels > 1:
                    for idx in range(n_channels, n_rows * n_cols):
                        axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(f'{paths}/other_channels_amplitude.png', dpi=300)
                print(f'Saved {paths}/other_channels_amplitude.png')
                plt.close()
                
        '''
                
                fig, axes, n_rows, n_cols = create_subplot_layout(n_channels)
                
                fig.suptitle(f"Other Channels Waveforms - Run {run_index}")
                
                for idx, name in enumerate(other_channels):
                    if len(hist_wf.get(name, [])) > 0:
                        axes[idx].set_title(f"{name} Waveforms")
                        axes[idx].set_xlabel("Time (ns)")

                        x_data, y_data = zip(*hist_wf[name])
                    
                        scatter = axes[idx].scatter(x_data, y_data, alpha=0.6, s=0.1)
                        axes[idx].set_ylabel("Amplitude (mV)")
                        axes[idx].set_xlabel("Time (ns)")

                    else:
                        axes[idx].text(0.5, 0.5, f"No data for {name}", ha='center', va='center', transform=axes[idx].transAxes)
                
                if n_channels > 1:
                    for idx in range(n_channels, n_rows * n_cols):
                        axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(f'{paths}/other_channels_run{run_index}.png', dpi=300)
                print(f'Saved {paths}/other_channels_run{run_index}.png')
                plt.close()
            
                
            else:
                print(f"No waveform data collected for run {run_index}")

        '''