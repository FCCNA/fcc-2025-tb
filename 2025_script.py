# %%
import json
import argparse
import functions
import os
# %%



# %%
parser = argparse.ArgumentParser(description='Process MIDAS runs')
parser.add_argument('--runs', type=str, nargs='+', required=True, 
                    help='Run indices to process (integers or ranges like 202:205)')
parser.add_argument('--path', type=str, 
                    help='Path to the run files, default = /eos/experiment/drdcalo/maxicc/TBCERN_24Sept2025_vx2730/', 
                    default='/eos/experiment/drdcalo/maxicc/TBCERN_24Sept2025_vx2730/')
parser.add_argument('--plot_check', action='store_true', 
                    help='Enable plotting checks')
parser.add_argument('--not_save_wf', action='store_false', default=True, 
                    help='Enable saving waveforms')
parser.add_argument('--json', default = 'config_fase1.json', type = str,
                    help='Select JSON for configuration')
parser.add_argument('--fast_check', action='store_true',
                    help='Enable fast check mode')
parser.add_argument('--write', action='store_true',
                    help='Create the dataframe the file')
parser.add_argument('--pathmib', action='store_true',
                    help='save files to lucchini path')
args = parser.parse_args()

# Load JSON file
with open(args.json, 'r') as f:
    json_data = json.load(f)

path = args.path
plot_check = (args.plot_check or args.fast_check)
fast_check = (args.fast_check)
sw = args.not_save_wf
run_indices = []
for arg in args.runs:
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
                            fc = fast_check,
                            json_data = json_data, 
                            save_waveforms = sw,
                            write = args.write,
                            pathmib = args.pathmib)
    if args.not_save_wf:
        os.system(f'python3 example_analysis.py --run {run_index}')

