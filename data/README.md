# data/ — Market Data

## Structure
```
data/
├── crypto/
│   ├── fetch_binance_lob.py
│   ├── configs/
│   │   └── binance_params.yaml
│   └── raw/                          ← gitignored
├── lobster/                          ← gitignored when data arrives
│   └── README.md                     ← instructions for obtaining LOBSTER data
├── processed/
│   └── lob_snapshots.npy             ← gitignored, AE pre-training input
└── process_lobster.py                ← handles both sources via --data_dir
```

This project's own synthetic-LOBSTER-data generator and the background-agent
calibration pipeline it fed (`data/synthetic/`, `data/calibration/`) were
archived — those calibrated parameters were never consumed by the live
simulator, which uses ABIDES's own `rmsc04` background config unmodified.
See `archive/lobster_calibration/` if reviving that pipeline.

## LOBSTER Data
LOBSTER data must be obtained separately from lobsterdata.com.
Place raw files in `data/lobster/` — these are gitignored.
Run `python data/process_lobster.py` to generate processed snapshots.
