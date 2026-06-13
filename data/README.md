# data/ — Market Data and Calibration

## Structure
```
data/
├── lobster/          # Raw LOBSTER tick data (gitignored — too large)
│   └── README.md     # Instructions for obtaining LOBSTER data
├── calibration/      # Fitted background agent parameters
│   ├── hawkes_params.json
│   └── agent_params.json
└── processed/        # Preprocessed LOB snapshots for AE pre-training
```

## LOBSTER Data
LOBSTER data must be obtained separately from lobsterdata.com.
Place raw files in `data/lobster/` — these are gitignored.
Run `python data/process_lobster.py` to generate processed snapshots.
