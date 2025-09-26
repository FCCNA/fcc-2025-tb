#!/bin/bash

# Script per eseguire rsync ogni 15 minuti
while true; do
    echo "$(date): Avvio rsync..."
    rsync -avh --progress fcc@128.141.41.208:/data/runs/*mid.gz /eos/experiment/drdcalo/maxicc/TBCERN_24Sept2025_vx2730/
    echo "$(date): Rsync completato. Attendo 15 minuti..."
    sleep 600  # 900 secondi = 15 minuti
done
