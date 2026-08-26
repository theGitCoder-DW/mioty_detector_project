"""Thin command-line wiring layer."""
import argparse
from pathlib import Path
import numpy as np
from .config import *
from .io import load_iq_file
from .pilot import generate_pilot_reference
from .detector import (adaptive_threshold, correlate_single_frequency, find_bursts_2d,
                       generate_frequency_grid, generate_frequency_shifted_references,
                       strongest_detection)
from .correlation.cpu import CPUCorrelationBackend # to support CPU execution
from .correlation.cuda import CUDACorrelationBackend # to support cuda-config
from .plots import (get_pyplot, plot_reference_waveform, plot_time_frequency_correlation,
                    compute_local_correlation_surface, plot_local_correlation_surface)

def build_parser():
    p=argparse.ArgumentParser(description="MIOTY/TS-UNB pilot burst detector")
    p.add_argument("iq_path",nargs="?",default=None)
    p.add_argument("--threshold",type=float,default=None)
    p.add_argument("--pilot",default=PILOT_BITS)
    p.add_argument("--oversampling",type=int,default=OVERSAMPLING)
    p.add_argument("--sample-rate",type=float,default=None)
    p.add_argument("--plot",action="store_true")
    p.add_argument("--save-plots",default=None)
    p.add_argument("--freq-search",action="store_true")
    p.add_argument("--num-channels",type=int,default=NUM_EU1_CHANNELS)
    p.add_argument("--freq-step",type=float,default=FREQUENCY_SEARCH_STEP_HZ,help=(
        "Frequency spacing for future fine-frequency refinement "
        "around detected peaks."
    ),)
    p.add_argument("--backend",choices=["cpu", "cuda"],default="cpu",
    help=(
        "Correlation backend. "
        "Use 'cpu' for SciPy or 'cuda' for CuPy/GPU. "
        "Default: cpu."
    ),)

    return p

def self_test():
    ref=generate_pilot_reference(); rng=np.random.default_rng(0); offset=1234
    received=((rng.normal(size=5000)+1j*rng.normal(size=5000))*0.05).astype(np.complex64)
    received[offset:offset+len(ref)]+=ref
    corr=correlate_single_frequency(received,ref); detected=int(np.argmax(corr))
    print(f"True offset: {offset}\nDetected offset: {detected}\nPeak magnitude: {corr[detected]:.4f}")
    assert detected==offset

def run_frequency_search(args,received):
    if args.sample_rate is None: raise ValueError("--freq-search requires --sample-rate")
    freqs=generate_frequency_grid(
    num_channels=args.num_channels,
    channel_spacing_hz=CHANNEL_SPACING_HZ,)

    freqs,refs=generate_frequency_shifted_references(args.pilot,args.sample_rate,freqs,args.oversampling)
    # FINAL CUDA MILESTONE: replace this backend selection only.
    backend = create_correlation_backend(
    args.backend)
    surface=backend.correlate(received,refs)
    threshold=args.threshold if args.threshold is not None else adaptive_threshold(surface)
    detections=find_bursts_2d(freqs,surface,threshold,pilot_bits=args.pilot,oversampling=args.oversampling)
    strongest=strongest_detection(detections)
    print(f"Frequency hypotheses: {len(freqs)}")
    print(f"Correlation surface: {surface.shape}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Detections: {len(detections)}")
    if args.plot or args.save_plots:
        plt=get_pyplot(args.plot)
        fig=plot_time_frequency_correlation(freqs,surface,detections,strongest,args.sample_rate,plt,args.plot)
        local=None
        if strongest:
            ref=generate_pilot_reference(args.pilot,args.oversampling)
            tau,df,db,ps,pf,pm=compute_local_correlation_surface(received,ref,args.sample_rate,strongest.sample_index,strongest.frequency_hz,frequency_step_hz=args.freq_step)
            local=plot_local_correlation_surface(tau,df,db,plt,args.plot)
            print(f"Strongest local maximum: sample={ps}, frequency={pf:+.2f} Hz, magnitude={pm:.4f}")
        if args.save_plots:
            out=Path(args.save_plots); out.mkdir(parents=True,exist_ok=True)
            fig.savefig(out/"04_time_frequency_correlation.png",dpi=150,bbox_inches="tight")
            if local: local.savefig(out/"05_local_global_max_correlation.png",dpi=150,bbox_inches="tight")
        if args.plot: plt.show()

def run_single_frequency(args,received):
    from scipy.signal import find_peaks
    ref=generate_pilot_reference(args.pilot,args.oversampling); corr=correlate_single_frequency(received,ref)
    median=np.median(corr); mad=np.median(np.abs(corr-median)); threshold=args.threshold if args.threshold is not None else median+8*1.4826*mad
    idx,_=find_peaks(corr,height=threshold,distance=len(ref)); print(f"Threshold: {threshold:.4f}\nDetected {len(idx)} burst(s).")
    if args.plot or args.save_plots:
        plt=get_pyplot(args.plot); fig=plot_reference_waveform(ref,args.pilot,args.oversampling,plt,args.plot)
        from types import SimpleNamespace
        detections=[SimpleNamespace(sample_index=int(i),frequency_hz=0.0,magnitude=float(corr[i])) for i in idx]
        # Keep single-frequency plotting intentionally simple.
        if args.save_plots:
            out=Path(args.save_plots); out.mkdir(parents=True,exist_ok=True); fig.savefig(out/"01_reference_waveform.png",dpi=150,bbox_inches="tight")
        if args.plot: plt.show()

def create_correlation_backend(
    backend_name,
):
    """
    Create the requested correlation backend.

    Supported:
        cpu
        cuda
    """

    backend_name = backend_name.lower()

    if backend_name == "cpu":
        print(
            "Using CPU correlation backend."
        )
        return CPUCorrelationBackend()

    if backend_name == "cuda":
        print(
            "Using CUDA correlation backend."
        )
        return CUDACorrelationBackend()

    raise ValueError(
        f"Unknown backend: {backend_name}. "
        f"Choose 'cpu' or 'cuda'."
    )

def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.iq_path is None: return self_test()
    received=load_iq_file(args.iq_path); print(f"Loaded '{args.iq_path}': {len(received)} IQ samples")
    run_frequency_search(args,received) if args.freq_search else run_single_frequency(args,received)
