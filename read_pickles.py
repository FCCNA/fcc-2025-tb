# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
          'font.size': 20,
          'figure.figsize': (16, 9),
          'axes.grid': False,
          'grid.linestyle': '-',
          'grid.alpha': 0.2,
          'lines.markersize': 5.0,
          'xtick.minor.visible': True,
          'xtick.direction': 'in',
          'xtick.major.size': 10.0,
          'xtick.minor.size': 5.0,
          'xtick.top': True,
          'ytick.minor.visible': True,
          'ytick.direction': 'in',
          'ytick.major.size': 10.0,
          'ytick.minor.size': 5.0,
          'ytick.right': True,
          'errorbar.capsize': 0.0,
          'figure.max_open_warning': 50,
})
from cycler import cycler  # Permette di impostare cicli di colori

# Ottieni i colori dalla palette Pastel1
pastel_colors = plt.get_cmap("Set2").colors

# Imposta lo stile dei colori per tutti i plot
plt.rcParams["axes.prop_cycle"] = cycler(color=pastel_colors)
from scipy.optimize import curve_fit
# %% Load the DataFrame

import __template__new as mcj


# %%

df = pd.read_pickle('pickles/run692_wf.pkl')
# %% Explore the DataFrame
# %%

df.info() 
for detector in df.columns.levels[0]:
    print(detector, list(df[detector].keys()))

# %%

plt.plot(df['Cherenkov']['WF'][0])


def myfunc(x, c, s, t0, of):
    return mcj.wf_function(x, interS_template, interC_template, c, s, t0, of)

times = np.arange(0, )
dt = times[1]-times[0]
ranger = [int(times.min()), int(times.max())]
npoints = len(times)
interS_template, interC_template, bins = mcj.get_templates(crystal,
                                                        sipm,
                                                        SPR,
                                                        ranger,
                                                        dt,
                                                        nsimS=1E7,
                                                        nsimC=1E7,
                                                        normalize=True,
                                                        graphics=False,
                                                        Laser_Calib = laser, 
                                                        run = log_laser[sipm][ampl])
output = []

print(f'Fitting Channel {ch}...')
# Get the waveform data
x_data = np.arange(len(df['Cherenkov']['WF'][0]))
y_data = df['Cherenkov']['WF'][0]

# Initial parameter guess [a, tf, tr, x0, c, dt, fc]
initial_guess = [np.max(y_data), 100, 10, np.argmax(y_data), np.min(y_data), 1.0, 0.01]

# Perform the fit
popt, pcov = curve_fit(db_exp_hp, x_data, y_data, p0=initial_guess)

# Plot the fit
plt.plot(x_data, db_exp_hp(x_data, *popt), 'r--', label=f'Fit: a={popt[0]:.2f}, tf={popt[1]:.2f}, tr={popt[2]:.2f}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Cherenkov Waveform with Double Exponential + Highpass Fit')
# %%



# %%
