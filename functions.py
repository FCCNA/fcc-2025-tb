import numpy as np
from tqdm import tqdm 
import midas.file_reader
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


def convert(wf, altri_parametri = 1):
    return wf * altri_parametri

def read_waveform(run_index, path, om, json_data, save_waveforms):
    run = f'run{run_index:05d}'
    mfile = midas.file_reader.MidasFile(path + run + '.mid.gz', use_numpy=True)

    try:
        # Try to find the special midas event that contains an ODB dump.
        odb = mfile.get_bor_odb_dump()
        odb_dict = odb.data;

        # The full ODB is stored as a nested dict withing the `odb.data` member.
        run_number = odb_dict["Runinfo"]["Run number"]
        print("We are looking at a file from run number %s" % run_number)
    except RuntimeError:
        # No ODB dump found (mlogger was probably configured to not dump
        # the ODB at the start of each subrun).
        print("No begin-of-run ODB dump found")


    mfile.jump_to_start()

    print("Reading ODB")
    adctovolt = {}
    #Channel_Enabled = odb_dict['Equipment']['Trigger']['Settings']['Channel Enable']
    #scale = odb_dict['Equipment']['Trigger']['Settings']['Channel Scale']
    #position = odb_dict['Equipment']['Trigger']['Settings']['Channel Position']
    #HPOS = odb_dict['Equipment']['Trigger']['Settings']['Horizontal Position']
    #SARA = odb_dict['Equipment']['Trigger']['Settings']['Sample Rate']
    for channel_id in json_data["Channels"]:
        adctovolt[channel_id] = odb_dict["Equipment"]["FeLibFrontend"]["Settings"][f"Channel{channel_id}"]["adctovolts"]
    total_events = sum(1 for _ in mfile if not _.header.is_midas_internal_event() and _.header.event_id == 1)
    mfile.jump_to_start()

    bit = 2**16
    
    new_data_list = []

    progress_bar = tqdm(total=total_events, colour='green', desc="Processing events")
    progress_bar.set_postfix(evts=total_events)
    data_for_df = []
    if om:
        hist_wf = {}
        hist_wf["Cherenkov"] = []
        hist_wf["Scintillator"] = []
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
            wf = convert(bank.data, adctovolt[str(bank_index)])
            wf_mediata = media_mobile(wf, window_size=20)
            #print(wf)
            if om:
                if name in ['Cherenkov', 'Scintillator']:
                    hist_wf[name].extend((i, sample) for i, sample in enumerate(wf))
                
            ampl = np.max(np.abs(wf))
            ampl_mediata = np.max(np.abs(wf_mediata))

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

        event_data[('Generic_Info', 'Times')] = np.ones((len(wf)))
        
        data_for_df.append(event_data)

    progress_bar.close()

    os.makedirs('pickles', exist_ok=True)
    output_df = pd.DataFrame(data_for_df)
    output_df.columns = pd.MultiIndex.from_tuples(output_df.columns)
    add = '_wf' if save_waveforms else ''
    output_df.to_pickle(f'pickles/run{run_index}{add}.pkl')
    print(f"DataFrame for run {run_index} saved with {output_df.shape[0]} events.")

    if om:
        print("Saving Plots for Checks")
        os.makedirs('check_plot', exist_ok=True)
        for channel_id in json_data["Channels"]:
            name = json_data["Channels"][channel_id]["name"]
            plt.figure(figsize=(10, 6))
            plt.hist(output_df[(name, 'Amplitude')], bins = 100, alpha=0.7)
            plt.title(f"{name} Amplitude Distribution - Run {run_index}")
            plt.xlabel("Amplitude")
            plt.ylabel("Counts")
            plt.tight_layout()
            plt.savefig(f'check_plot/{name}_amplitude_run{run_index}.png', dpi = 300)
            plt.close()
            if name in ['Cherenkov', 'Scintillator']:
                if len(hist_wf[name]) > 0:
                    plt.figure(figsize=(10, 6))
                    plt.title(f"{name} Waveforms - Run {run_index}")
                    plt.xlabel("Time (samples)")
                    up = json_data['Channels'].get(channel_id, {}).get('Upper_Limit', json_data["Upper_Limit"])
                    low = json_data['Channels'].get(channel_id, {}).get('Lower_Limit', json_data["Lower_Limit"])
                    plt.axhline(up, color='black', linewidth=0.8)
                    plt.axhline(low, color='black', linewidth=0.8)
                    x_bins = np.max([i for i, _ in hist_wf[name]]) + 1
                    plt.hist2d(*zip(*hist_wf[name]), cmap='viridis', bins=[x_bins, 200])
                    plt.colorbar(label='Counts')
                    plt.tight_layout()
                    plt.savefig(f'check_plot/{name}_run{run_index}.png', dpi = 300)
                else:
                    print(f"No waveform data collected for run {run_index}")

