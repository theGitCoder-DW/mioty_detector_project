"""All plotting code. Signal-processing modules stay free of matplotlib."""
import numpy as np

def get_pyplot(interactive=True):
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def plot_reference_waveform(reference, pilot_bits, oversampling, plt=None, interactive=True):
    if plt is None: plt = get_pyplot(interactive)
    n = np.arange(len(reference)); phase = np.unwrap(np.angle(reference))
    step = np.diff(phase, prepend=phase[0])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(n, reference.real, label="I"); ax[0].plot(n, reference.imag, label="Q")
    ax[0].set_ylabel("Amplitude"); ax[0].set_title(f"Pilot reference: {pilot_bits}"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(n, step); ax[1].set_ylabel("Phase step [rad/sample]"); ax[1].set_xlabel("Sample index"); ax[1].grid(alpha=.3)
    for i, bit in enumerate(pilot_bits):
        x=i*oversampling
        for a in ax: a.axvline(x, linestyle=":", alpha=.35)
        ax[0].text(x+oversampling/2, ax[0].get_ylim()[1]*.85, bit, ha="center", fontsize=9)
    fig.tight_layout(); return fig

def plot_time_frequency_correlation(frequency_hz, surface, detections=None, strongest=None,
                                     sample_rate=None, plt=None, interactive=True):
    """Full-recording frequency-vs-time map with only the strongest markers."""
    if plt is None: plt=get_pyplot(interactive)
    n_time=surface.shape[1]
    time_ms=np.arange(n_time)/sample_rate*1000 if sample_rate else np.arange(n_time)
    fig, ax=plt.subplots(figsize=(13,6))
    im=ax.imshow(surface, aspect="auto", origin="lower",
                 extent=[time_ms[0], time_ms[-1], frequency_hz[0]/1000, frequency_hz[-1]/1000],
                 cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Correlation magnitude")
    if detections:
        for rank, d in enumerate(sorted(detections,key=lambda x:x.magnitude,reverse=True)[:24],1):
            x=d.sample_index/sample_rate*1000 if sample_rate else d.sample_index
            ax.scatter(x,d.frequency_hz/1000,s=60,facecolors="none",edgecolors="cyan",linewidths=1.5,zorder=5)
            ax.annotate(str(rank),(x,d.frequency_hz/1000),xytext=(5,5),textcoords="offset points",fontsize=8)
    if strongest is not None:
        x=strongest.sample_index/sample_rate*1000 if sample_rate else strongest.sample_index
        ax.scatter(x,strongest.frequency_hz/1000,s=180,marker="*",facecolors="magenta",edgecolors="black",zorder=10,label="Global maximum")
        ax.legend()
    ax.set_xlabel("Time [ms]" if sample_rate else "Sample index"); ax.set_ylabel("Frequency offset [kHz]")
    ax.set_title("MIOTY time-frequency pilot correlation"); ax.grid(alpha=.2); fig.tight_layout(); return fig

def compute_local_correlation_surface(received_iq, reference_iq, sample_rate, center_sample,
                                       center_frequency_hz, time_half_width_ms=10,
                                       frequency_half_width_hz=1000, frequency_step_hz=25):
    """Compute C(tau,f) around the strongest burst and normalize max to 0 dB."""
    from scipy.signal import correlate
    half=int(round(time_half_width_ms*sample_rate/1000)); start=max(0,int(center_sample)-half)
    end=min(len(received_iq),int(center_sample)+half+len(reference_iq)); segment=received_iq[start:end]
    freqs=np.arange(center_frequency_hz-frequency_half_width_hz, center_frequency_hz+frequency_half_width_hz+frequency_step_hz/2, frequency_step_hz)
    t=np.arange(len(reference_iq))/sample_rate; energy=np.sum(np.abs(reference_iq)**2)
    surface=np.empty((len(freqs),len(segment)-len(reference_iq)+1),dtype=np.float64)
    for row,f in enumerate(freqs):
        shifted=reference_iq*np.exp(1j*2*np.pi*f*t)
        c=correlate(segment,shifted,mode="valid",method="fft")
        surface[row]=np.abs(c)/np.sqrt(energy)
    r,c=np.unravel_index(np.argmax(surface),surface.shape); maximum=surface[r,c]
    peak_f=freqs[r]; peak_sample=start+c
    tau=(np.arange(surface.shape[1])-c)/sample_rate*1000
    df=freqs-peak_f
    db=np.maximum(20*np.log10(np.maximum(surface/maximum,1e-12)),-21)
    return tau,df,db,peak_sample,peak_f,maximum

def plot_local_correlation_surface(time_offsets_ms, frequency_offsets_hz, surface_db, plt=None, interactive=True):
    """Reference-style local correlation lobe centered at (0 ms, 0 Hz)."""
    if plt is None: plt=get_pyplot(interactive)
    levels=np.arange(-21,0.1,3)
    fig,ax=plt.subplots(figsize=(10,7))
    cf=ax.contourf(time_offsets_ms,frequency_offsets_hz/1000,surface_db,levels=levels,cmap="viridis",extend="min")
    fig.colorbar(cf,ax=ax,ticks=[-21,-18,-15,-12,-9,-6,-3,0],label="Normalized correlation [dB]")
    ax.scatter(0,0,s=35,c="black",zorder=10); ax.axvline(0,linewidth=.8,alpha=.5); ax.axhline(0,linewidth=.8,alpha=.5)
    ax.set_xlabel("Time offset [ms]"); ax.set_ylabel("Frequency offset [kHz]"); ax.set_title("Local 2-D pilot correlation around strongest burst"); ax.grid(alpha=.2)
    fig.tight_layout(); return fig
