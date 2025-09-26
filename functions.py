import numpy as np
from tqdm import tqdm 
import midas.file_reader
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

def create_beam_position_plot(delta_x, delta_y, title, filename, figsize=(10, 8), bins=250, range_vals=(-40, 40)):

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
    
def read_waveform(run_index, path, om, fc, json_data, read_waveforms, write, pathmib):
    run = f'run{run_index:05d}'
    try:
        mfile = midas.file_reader.MidasFile(path + run + '.mid.gz', use_numpy=True)
    except FileNotFoundError:
        print(f"File {path + run + '.mid.gz'} not found.")
        return None, None, None
    paths = f'check_plot/{run_index}'
    os.makedirs(f'check_plot/{run_index}', exist_ok=True)
    with open('logbook.yaml', 'r') as file:
        logbook_data = yaml.safe_load(file)

    if run_index in logbook_data:
        logbook_entry = logbook_data[run_index]
    else:
        logbook_entry = {}
        logbook_entry["particle"] = ''
        logbook_entry["angle"] = 0
        logbook_entry['crystal'] = ''

    if pathmib:
        op = '/eos/experiment/drdcalo/maxicc/TBCERN_24Sept2025_vx2730/'
    else: 
        op = './'
    if write:

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
        
        print("ODB complete")
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
            event_data['Run_Number'] = run_number
            event_data['Event'] = event.header.serial_number

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
                event_data[f'{name}_pedestal'] = piedistallo
                wf = convert(bank.data, yaml_output[name]["adctovolts"], piedistallo)
                event_data[f'{name}_WF'] = wf

                if is_saturato(bank.data, wf, bit = bit):
                    satur[name] += 1
                    event_data[f'{name}_is_satur'] = True
                else:
                    event_data[f'{name}_is_satur'] = False
                

                ampl = np.max(np.abs(wf))
                event_data[f'{name}_amplitude'] = ampl
                t0 = find_t0(wf, ampl*0.1, times)
                event_data[f'{name}_t0'] = t0
                event_data[f'{name}_tmax'] = times[np.argmax(np.abs(wf))]
                

                if name in ["Scintillator", "Cherenkov"]:
                    wf_mediata = media_mobile(wf, window_size=20)
                    ampl_mediata = np.max(np.abs(wf_mediata))
                    event_data[f'{name}_amplitude_mediata'] = ampl_mediata
                    event_data[f'{name}_charge'] = np.sum(wf)


            
            data_for_df.append(event_data)

        progress_bar.close()
    
        output_df = pd.DataFrame(data_for_df)
        #output_df.columns = pd.MultiIndex.from_tuples(output_df.columns)
        conversion = 0.18
        output_df['wc_x'] = conversion * (output_df['Chamber_x_1_t0'] - output_df['Chamber_x_2_t0'])
        output_df['wc_y'] = conversion * (output_df['Chamber_y_1_t0'] - output_df['Chamber_y_2_t0'])
        os.makedirs(f'{op}parquet', exist_ok=True)
        # Write yaml_output to file
        yaml_filename = f'{op}yamls/run{run_index}_info.yaml'
        os.makedirs(f'{op}yamls', exist_ok=True)
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(yaml_output, yaml_file)
        print(f"YAML info for run {run_index} saved to {yaml_filename}")
        output_df.to_parquet(f'{op}parquet/run{run_index}_wf.parq')
        print(f"DataFrame for run {run_index} saved with {len(output_df)} events.")
        '''
        output_df_no_wf = output_df.copy()

        columns_to_remove = []
        for col in output_df_no_wf.columns:
            if len(col) > 1 and col[1] == 'WF':
                columns_to_remove.append(col)

        output_df_no_wf = output_df_no_wf.drop(columns=columns_to_remove)

        # Save the DataFrame without waveforms
        output_df_no_wf.to_parquet(f'{op}parquets/run{run_index}.pkl')
        print(f"DataFrame without waveforms for run {run_index} saved with {output_df_no_wf.shape[0]} events.")
        '''
        #print(f"Saturated events for run {run_index}: {satur}")
    else:
        print(f'Reading existing DataFrame {op}parquet/run{run_index}_wf.parq')

        col_list = var.col_list_no_wf
        output_df = pd.read_parquet(f'{op}parquet/run{run_index}_wf.parq', columns = col_list, engine = 'pyarrow')
        
        with open(f'{op}yamls/run{run_index}_info.yaml', 'r') as yaml_file:
            yaml_output = yaml.safe_load(yaml_file)
        satur = {}
        print(f"Loaded DataFrame for run {run_index} with {len(output_df)} events.")
        paths = f'check_plot/{run_index}'
        os.makedirs(f'check_plot/{run_index}', exist_ok=True)


    if not fc:
        if om:
            print("Creating Efficiency Plots")
            list_of_cuts_plastic = {
                'Trigger': None,
                'Plastico': (output_df[('Plastico_amplitude')] > 0.1),
                'Chambers': (
                    (output_df[('Chamber_x_1_amplitude')] > 0.5) & 
                    (output_df[('Chamber_x_2_amplitude')] > 0.5) &
                    (output_df[('Chamber_y_1_amplitude')] > 0.5) & 
                    (output_df[('Chamber_y_2_amplitude')] > 0.5)
                ),
            }
            create_efficiency_plot(output_df, list_of_cuts_plastic, run_index, paths, 'crystal', logbook_entry)
            list_of_cuts_chambers = {
                'Trigger': None,
                'Chambers': (
                    (output_df[('Chamber_x_1_amplitude')] > 0.5) & 
                    (output_df[('Chamber_x_2_amplitude')] > 0.5) &
                    (output_df[('Chamber_y_1_amplitude')] > 0.5) & 
                    (output_df[('Chamber_y_2_amplitude')] > 0.5)
                ),
            }
            create_efficiency_plot(output_df, list_of_cuts_chambers, run_index, paths, 'chambers', logbook_entry)

    if om:
        print("Saving Plots for Checks")

        if 'Chamber_x_1_amplitude' in output_df.columns and 'Chamber_x_2_amplitude' in output_df.columns and \
           'Chamber_y_1_amplitude' in output_df.columns and 'Chamber_y_2_amplitude' in output_df.columns:
            # Create canvas for Cherenkov and Scintillator
            cuts = ['chambers', 'plastico', 'crystal']

            fig, axes, n_rows, n_cols = create_subplot_layout(len(cuts))
            
            output_df_filtered = {}
            for cut in cuts:
                output_df_filtered[cut] = apply_data_cuts(output_df, cut, run_index)

            for idx, cut in enumerate(cuts):
                create_2d_histogram(axes[idx], output_df_filtered[cut]['wc_x'], output_df_filtered[cut]['wc_y'], f"{cuts[idx]} - {logbook_entry['particle']} - {logbook_entry['crystal']} - {logbook_entry['angle']}", run_index)

            plt.tight_layout()
            plt.savefig(f'{paths}/chambers_beamposition_both.png', dpi=300)
            print(f'Saved {paths}/chambers_beamposition_both.png')
            plt.close()

            for cut in cuts:
                create_beam_position_plot(output_df_filtered[cut]['wc_x'], output_df_filtered[cut]['wc_y'], 
                                        f"{cut} - {logbook_entry['particle']} - {logbook_entry['crystal']} - {logbook_entry['angle']}", 
                                        f'{paths}/chambers_beamposition_grid_{cut}.png')


        if 'Cherenkov_amplitude' in output_df.columns and 'Scintillator_amplitude' in output_df.columns:
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
            cuts = ['chambers', 'plastico', 'crystal']
            for cut in cuts:
                create_amplitude_histo(output_df_filtered[cut], run_index, paths, logbook_entry, suffix=cut)


        if not fc:
                
            other_channels = [name for channel_id in json_data["Channels"] 
                    for name in [json_data["Channels"][channel_id]["name"]] 
                    if name not in ['Cherenkov', 'Scintillator']]
            
            if other_channels:
                n_channels = len(other_channels)

                fig, axes, n_rows, n_cols = create_subplot_layout(n_channels)

                fig.suptitle(f"Other Channels Amplitudes - {logbook_entry['particle']} - {logbook_entry['crystal']} - {logbook_entry['angle']}")

                for idx, name in enumerate(other_channels):
                    if (f'{name}_amplitude') in output_df.columns:
                        axes[idx].hist(output_df[f'{name}_amplitude'], bins=100, alpha=0.7)
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