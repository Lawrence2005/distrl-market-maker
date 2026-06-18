# data/ — Market Data and Calibration

## Structure
```
data/
├── synthetic/
│   ├── generate_synthetic_lobster.py
│   ├── configs/
│   │   └── synthetic_params.yaml
│   └── generated/                    ← gitignored
├── crypto/
│   ├── fetch_binance_lob.py
│   ├── configs/
│   │   └── binance_params.yaml
│   └── raw/                          ← gitignored
├── lobster/                          ← gitignored when data arrives
│   └── README.md                     ← instructions for obtaining LOBSTER data
├── calibration/
│   ├── hawkes_params.json            ← populated from whichever source runs first
│   └── agent_params.json
├── processed/
│   └── lob_snapshots.npy             ← gitignored, AE pre-training input
└── process_lobster.py                ← handles all three sources via --data_dir
```

## LOBSTER Data
LOBSTER data must be obtained separately from lobsterdata.com.
Place raw files in `data/lobster/` — these are gitignored.
Run `python data/process_lobster.py` to generate processed snapshots.
