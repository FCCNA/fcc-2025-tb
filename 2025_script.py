# %%
import json
import argparse
import functions
# %%



# %%
parser = argparse.ArgumentParser(description='Process MIDAS runs')
parser.add_argument('--run_indices', type=str, nargs='+', required=True, 
                    help='Run indices to process (integers or ranges like 202:205)')
parser.add_argument('--path', type=str, 
                    help='Path to the run files, default = /eos/home-g/ggaudino/fcc-testbeam_analysis/nuovi_runs/', 
                    default='/eos/home-g/ggaudino/fcc-testbeam_analysis/nuovi_runs/')
parser.add_argument('--plot_check', action='store_true', 
                    help='Enable plotting checks')
parser.add_argument('--save_wf', action='store_true', 
                    help='Enable saving waveforms')
parser.add_argument('--json', default = 'config.json', type = str,
                    help=' Select JSON for configuration')
args = parser.parse_args()

# Load JSON file
with open(args.json, 'r') as f:
    json_data = json.load(f)

path = args.path
plot_check = args.plot_check
sw = args.save_wf
run_indices = []
for arg in args.run_indices:
    if ':' in arg:
        start, end = map(int, arg.split(':'))
        run_indices.extend(range(start, end + 1))
    else:
        run_indices.append(int(arg))


# %%
for run_index in run_indices:
    functions.read_waveform(run_index = run_index, 
                            path = path, 
                            om = plot_check,
                            json_data = json_data, 
                            save_waveforms = sw)


# %%
