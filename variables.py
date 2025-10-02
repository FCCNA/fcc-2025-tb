col_list = ['Run_Number', 'Event', 'PMT_pedestal', 'PMT_is_satur', 'PMT_amplitude', 'PMT_t0', 'PMT_tmax', 'Chamber_x_1_pedestal', 'Chamber_x_1_is_satur', 'Chamber_x_1_amplitude', 'Chamber_x_1_t0', 'Chamber_x_1_tmax', 'Chamber_x_2_pedestal', 'Chamber_x_2_is_satur', 'Chamber_x_2_amplitude', 'Chamber_x_2_t0', 'Chamber_x_2_tmax', 'Chamber_y_1_pedestal', 'Chamber_y_1_is_satur', 'Chamber_y_1_amplitude', 'Chamber_y_1_t0', 'Chamber_y_1_tmax', 'Chamber_y_2_pedestal', 'Chamber_y_2_is_satur', 'Chamber_y_2_amplitude', 'Chamber_y_2_t0', 'Chamber_y_2_tmax', 'Scintillator_pedestal', 'Scintillator_pedestal_std','Scintillator_is_satur', 'Scintillator_amplitude', 'Scintillator_t0', 'Scintillator_tmax', 'Scintillator_amplitude_mediata', 'Scintillator_charge', 'Scintillator_WF', 'Cherenkov_pedestal', 'Cherenkov_pedestal_std', 'Cherenkov_is_satur', 'Cherenkov_amplitude', 'Cherenkov_t0', 'Cherenkov_tmax', 'Cherenkov_amplitude_mediata', 'Cherenkov_charge', 'Cherenkov_WF', 'Plastico_pedestal', 'Plastico_is_satur', 'Plastico_amplitude', 'Plastico_t0', 'Plastico_tmax', 'wc_x', 'wc_y']

col_list_no_wf = ['Run_Number', 'Event', 'PMT_pedestal', 'PMT_is_satur', 'PMT_amplitude', 'PMT_t0', 'PMT_tmax', 'Chamber_x_1_pedestal', 'Chamber_x_1_is_satur', 'Chamber_x_1_amplitude', 'Chamber_x_1_t0', 'Chamber_x_1_tmax', 'Chamber_x_2_pedestal', 'Chamber_x_2_is_satur', 'Chamber_x_2_amplitude', 'Chamber_x_2_t0', 'Chamber_x_2_tmax', 'Chamber_y_1_pedestal', 'Chamber_y_1_is_satur', 'Chamber_y_1_amplitude', 'Chamber_y_1_t0', 'Chamber_y_1_tmax', 'Chamber_y_2_pedestal', 'Chamber_y_2_is_satur', 'Chamber_y_2_amplitude', 'Chamber_y_2_t0', 'Chamber_y_2_tmax', 'Scintillator_pedestal',  'Scintillator_pedestal_std', 'Scintillator_is_satur', 'Scintillator_amplitude', 'Scintillator_t0', 'Scintillator_tmax', 'Scintillator_amplitude_mediata', 'Scintillator_charge', 'Cherenkov_pedestal', 'Cherenkov_pedestal_std', 'Cherenkov_is_satur', 'Cherenkov_amplitude', 'Cherenkov_t0', 'Cherenkov_tmax', 'Cherenkov_amplitude_mediata', 'Cherenkov_charge', 'Plastico_pedestal', 'Plastico_is_satur', 'Plastico_amplitude', 'Plastico_t0', 'Plastico_tmax', 'wc_x', 'wc_y']


plots_info = {
    "amplitude": {
        "xlabel": "Amplitude [mV]",
        "title": "Amplitude Distribution"
    },
    "amplitude_mediata": {
        "xlabel": "Amplitude Mean [mV]",
        "title": "Amplitude Mean Distribution"
    },\
    "pedestal_std": {
        "xlabel": "Pedestal Std [mV]",
        "title": "Pedestal Std Distribution"
    },
    "charge": {
        "xlabel": "Charge [a.u.]",
        "title": "Charge Distribution"
    },
    "t0": {
        "xlabel": "Rising Time [ns]",
        "title": "Rising Time Distribution"
    },
    "s": {
        "xlabel": "Scintillation Yield",
        "title": "Scintillation Yield Distribution"
    },
    "c": {
        "xlabel": "Cherenkov Yield",
        "title": "Cherenkov Yield Distribution"
    },
    "c_over_s": {
        "xlabel": "Cherenkov/Scintillator Yield Ratio",
        "title": "Cherenkov/Scintillator Yield Ratio Distribution"
    }
}