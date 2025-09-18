# fcc-2025-tb

## Script for the 2025 Test Beam

Il file principale è `2025_script.py`. 
Legge i dati dai file di acquisizione e li elabora per l'analisi. Principalemnte converte i run in file pickle, leggibili poi con pandas. 
```bash
    python3 2025_script.py --run_indices 692 693:695 697
```
Il programma leggerà solo i canali presenti nel json di configurazione. Di default, legge `config.json`. Per cambiare usare il comando --json.
Upper Limit e Lower Limit sono numeri i limiti del digitizer. Quelli generali vanno bene per tutti, se per uno specifico canale, ho diversi limiti, inserirli nel canale.
```json
{
    "Upper_Limit":1000,
    "Lower_Limit":0,
    "Channels": {
        "16": {
            "name": "Cherenkov", 
            "Upper_Limit": 10500,
            "Lower_Limit": 8000
        },
        "15": {
            "name": "Scintillator"
        }
    }
}
```


All'interno del pickle ci sono diversi sotto-dataframe, uno per ogni output del digitizer + uno generico con `event_number` e `run_number`.
Per ogni waveform salvo:
```python
['Amplitude',  # (Ampiezza Massima)
'Amplitude_Mediata', #(ampiezza massima della media mobile)
'TMax', #(tempo dell'ampiezza massima)
'Charge', #(somma)
't0'] #(tempo al 10% dell'ampiezza massima)
```
C'è la possibile di inserire nel pickle tutta la waveform con il comando `--save_wf`.

## Express Monitor
Lanciando
```bash
    python3 2025_script.py --run_indices 692 --plot_check
```
salverà dei plot nella cartella `check_plot`, con l'istogramma delle ampiezze e l'istogramma di tutte le waveform. 
Mettendo invece 
```bash
    python3 2025_script.py --run_indices 692 --fast_check
```
salverà solo i plot di cherenkov e scintillatore (non altri canali) e non salverà il pickle.
