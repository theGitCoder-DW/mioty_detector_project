# MIOTY Telegram Burst Detector

## Structure

```text
mioty_detector_project/
├── main.py
├── mioty_pilot_2d_fft.py       # compatibility launcher
├── requirements.txt
├── pyproject.toml
├── README.md
├── src/mioty_detector/
│   ├── config.py               # constants only
│   ├── io.py                   # IQ loading only
│   ├── pilot.py                # pilot/reference generation
│   ├── detector.py             # detection and frequency-bank logic
│   ├── plots.py                # all visualisation
│   ├── cli.py                  # command-line wiring
│   └── correlation/
│       ├── base.py             # backend interface
│       ├── cpu.py              # current CPU/SciPy implementation
│       └── cuda.py             # final CUDA milestone
└── tests/
```

## Run

```bash
python main.py
```

```bash
python main.py Path_of_the_test_Data --freq-search --sample-rate 228515.616 --plot
```

The final CUDA milestone should replace `CPUCorrelationBackend` with
`CUDACorrelationBackend` in `cli.py`, or better, make backend selection a
command-line option. The CPU implementation remains the golden reference for
numerical validation.
