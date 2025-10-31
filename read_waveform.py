
import midas.file_reader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
import os
from functions import x_times, convert, find_t0, media_mobile



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
        op = '/eos/experiment/drdcalo/maxicc/TBCERN_24Sept2025_vx2730/eor/'
    else: 
        op = './'
    if write:

        try:
            # Try to find the special midas event that contains an ODB dump.
            odb = mfile.get_eor_odb_dump()
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

                piedistallo = bank.data[:pretrigger_sample-100].mean()
                event_data[f'{name}_pedestal'] = piedistallo
                wf = convert(bank.data, yaml_output[name]["adctovolts"], piedistallo)
                event_data[f'{name}_WF'] = wf
                std_V = wf[:pretrigger_sample-100].std()
                event_data[f'{name}_pedestal_std'] = std_V

#                if is_saturato(bank.data, wf, bit = bit):
#                  satur[name] += 1
#                    event_data[f'{name}_is_satur'] = True
#                else:
#                    event_data[f'{name}_is_satur'] = False
                

                ampl = np.max(np.abs(wf))
                event_data[f'{name}_amplitude'] = ampl
                t0 = find_t0(wf, ampl*0.1, times)
                event_data[f'{name}_t0'] = t0
                event_data[f'{name}_tmax'] = times[np.argmax(np.abs(wf))]
                event_data[f'{name}_goodwf'] = (ampl > 3 * std_V)
                

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
        print(f"DataFrame for run {run_index} saved with {len(output_df)} events on {op}parquet/run{run_index}_wf.parq.")
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
                                        f'{paths}/chambers_beamposition_grid_{cut}.png', verbose=True)


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
