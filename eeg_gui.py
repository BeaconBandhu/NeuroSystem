#!/usr/bin/env python3
"""
Neurosystem EEG Training Studio
BioAmp EXG Pill + ESP32 — Clean → Preprocess → Annotate → Export → Train
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json, os, csv, threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

try:
    from scipy import signal as sp
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

# ── Palette ─────────────────────────────────────────────────────────────────
BG  = "#0d1117"; BG2 = "#161b22"; BG3 = "#21262d"; BG4 = "#30363d"
ACC = "#58a6ff"; ACC2 = "#bc8cff"
FG  = "#e6edf3"; FG2 = "#8b949e"; FG3 = "#6e7681"
OK  = "#3fb950"; WARN = "#d29922"; ERR = "#f85149"
BCOLORS = {"delta":"#ff6b6b","theta":"#ffd93d","alpha":"#6bcb77","beta":"#4d96ff","gamma":"#c77dff"}
BANDS   = ["delta","theta","alpha","beta","gamma"]
BKEYS   = {"d":"delta","t":"theta","a":"alpha","b":"beta","g":"gamma"}
FS      = 256

# ── Global state ─────────────────────────────────────────────────────────────
sessions       = {}
active_session = {"key": None}

def get_active():
    return sessions.get(active_session["key"])

# ═══════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════
class EEGData:
    def __init__(self):
        self.filepath = None; self.fs = FS; self.session_start = None
        self.batches = []; self.elapsed_ms = []
        self.raw = None; self.cleaned = None; self.preproc = None
        self.art_mask = None
        self.band = {n: [] for n in BANDS}
        self.events = []
        self.clean_steps = []; self.preproc_steps = []

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.filepath = path
        self.fs = d.get("sample_rate_hz", FS)
        try:
            self.session_start = datetime.fromisoformat(
                d.get("session_start_iso","").replace("Z","+00:00"))
        except Exception:
            self.session_start = None
        self.batches = d.get("batches", [])
        self._parse()
        self.reset_pipeline()

    def _parse(self):
        self.elapsed_ms = []; samps = []
        self.band = {n: [] for n in BANDS}
        for b in self.batches:
            self.elapsed_ms.append(b.get("elapsed_ms", 0))
            samps.extend(v / 10.0 for v in b.get("data_int16_x10", []))
            bp = b.get("band_power", {})
            for k, n in BKEYS.items():
                self.band[n].append(float(bp.get(k, 0.0)))
        self.raw = np.array(samps, dtype=np.float32)

    def reset_pipeline(self):
        self.cleaned  = self.raw.copy()
        self.preproc  = self.raw.copy()
        self.art_mask = np.zeros(len(self.raw), dtype=bool)
        self.clean_steps.clear(); self.preproc_steps.clear()

    @property
    def duration_s(self): return self.elapsed_ms[-1]/1000 if self.elapsed_ms else 0
    @property
    def n(self): return len(self.raw) if self.raw is not None else 0
    @property
    def t(self): return np.arange(self.n) / self.fs
    @property
    def bt(self): return np.array(self.elapsed_ms) / 1000
    def band_arr(self, name): return np.array(self.band[name])

    def dominant_at(self, elapsed_s):
        if not self.elapsed_ms: return "unknown"
        i = int(np.argmin(np.abs(self.bt - elapsed_s)))
        return max(BANDS, key=lambda n: self.band[n][i])

    def artifact_pct(self):
        return float(self.art_mask.mean() * 100) if self.art_mask is not None else 0.0

    def summary(self):
        if self.raw is None: return {}
        return {
            "file": os.path.basename(self.filepath or ""),
            "session_start": str(self.session_start),
            "duration_s": round(self.duration_s, 1),
            "total_samples": self.n,
            "sample_rate_hz": self.fs,
            "raw_uV": {"min": round(float(self.raw.min()),2),
                       "max": round(float(self.raw.max()),2),
                       "mean": round(float(self.raw.mean()),2),
                       "std":  round(float(self.raw.std()),2)},
            "band_power_means": {n: round(float(np.mean(self.band[n])),1) for n in BANDS},
            "artifact_pct": round(self.artifact_pct(), 2),
            "events":         self.events,
            "clean_steps":    self.clean_steps,
            "preproc_steps":  self.preproc_steps,
        }

# ═══════════════════════════════════════════════════════════════════════════
# Signal processor
# ═══════════════════════════════════════════════════════════════════════════
class Proc:
    @staticmethod
    def amp_mask(sig, thr): return np.abs(sig) > thr

    @staticmethod
    def iqr_mask(sig, mult=3.0):
        q1, q3 = np.percentile(sig, [25, 75]); iqr = q3 - q1
        return (sig < q1 - mult*iqr) | (sig > q3 + mult*iqr)

    @staticmethod
    def band_mask(band_arr, mult, spb=128):
        med = np.median(band_arr)
        if med == 0: return np.zeros(len(band_arr)*spb, bool)
        return np.repeat(band_arr > mult * med, spb)

    @staticmethod
    def interp(sig, mask):
        if not mask.any(): return sig.copy()
        out = sig.copy(); idx = np.arange(len(sig)); good = ~mask
        if good.sum() < 2: return out
        out[mask] = np.interp(idx[mask], idx[good], sig[good])
        return out

    @staticmethod
    def bandpass(sig, lo, hi, fs, order=4):
        if not SCIPY_OK: return sig
        nyq = fs/2; b, a = sp.butter(order, [lo/nyq, hi/nyq], btype="band")
        return sp.filtfilt(b, a, sig).astype(np.float32)

    @staticmethod
    def notch(sig, f0, fs, Q=30):
        if not SCIPY_OK: return sig
        b, a = sp.iirnotch(f0/(fs/2), Q)
        return sp.filtfilt(b, a, sig).astype(np.float32)

    @staticmethod
    def detrend(sig):
        if SCIPY_OK: return sp.detrend(sig).astype(np.float32)
        return (sig - sig.mean()).astype(np.float32)

    @staticmethod
    def zscore(sig):
        s = sig.std()
        return ((sig - sig.mean())/s).astype(np.float32) if s > 0 else sig

    @staticmethod
    def minmax(sig):
        r = sig.max() - sig.min()
        return ((sig - sig.min())/r).astype(np.float32) if r > 0 else sig

# ═══════════════════════════════════════════════════════════════════════════
# Widget helpers
# ═══════════════════════════════════════════════════════════════════════════
def lbl(p, text="", fg=FG, bg=BG2, size=9, bold=False, **kw):
    return tk.Label(p, text=text, fg=fg, bg=bg,
                    font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

def btn(p, text, cmd, color=ACC, fg=BG, **kw):
    return tk.Button(p, text=text, command=cmd, bg=color, fg=fg,
                     activebackground=color, activeforeground=fg,
                     relief=tk.FLAT, padx=10, pady=5,
                     font=("Segoe UI", 9, "bold"), cursor="hand2", **kw)

def entry(p, **kw):
    return tk.Entry(p, bg=BG3, fg=FG, insertbackground=FG,
                    relief=tk.FLAT, font=("Segoe UI", 9), **kw)

def dark_fig(w=10, h=5, rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h),
                             facecolor=BG2, constrained_layout=True)
    ax_list = axes.flat if hasattr(axes, "flat") else [axes]
    for ax in ax_list:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG2)
        ax.xaxis.label.set_color(FG2); ax.yaxis.label.set_color(FG2)
        ax.title.set_color(FG)
        for sp in ax.spines.values(): sp.set_edgecolor(BG4)
    return fig, axes

def embed_fig(fig, parent):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    tb.update(); tb.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas

def make_scale(p, label, var, from_, to, res, color=FG2, length=200):
    fr = tk.Frame(p, bg=BG2); fr.pack(fill=tk.X, padx=8, pady=3)
    lbl(fr, label, fg=color, bg=BG2, size=9).pack(side=tk.LEFT)
    tk.Scale(fr, from_=from_, to=to, resolution=res, variable=var,
             orient=tk.HORIZONTAL, bg=BG2, fg=FG, troughcolor=BG3,
             highlightthickness=0, length=length, showvalue=True,
             activebackground=ACC).pack(side=tk.LEFT, padx=6)
    return fr

def make_check(p, label, var, color=FG2):
    return tk.Checkbutton(p, text=label, variable=var, bg=BG2, fg=color,
                          selectcolor=BG3, activebackground=BG2,
                          font=("Segoe UI", 9))

# ═══════════════════════════════════════════════════════════════════════════
# Tab: Overview
# ═══════════════════════════════════════════════════════════════════════════
class OverviewTab(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG2)
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=BG2); top.pack(fill=tk.X, padx=14, pady=10)
        lbl(top, "Session Overview", fg=ACC, bg=BG2, size=12, bold=True).pack(anchor="w")
        self.info_var = tk.StringVar(value="No session loaded — use sidebar to load a file.")
        lbl(top, "", fg=FG2, bg=BG2, size=9, textvariable=self.info_var).pack(anchor="w", pady=3)

        # Band power cards row
        self.cards_frame = tk.Frame(self, bg=BG2)
        self.cards_frame.pack(fill=tk.X, padx=14, pady=4)
        self.band_vars = {}
        for name in BANDS:
            c = tk.Frame(self.cards_frame, bg=BG3, padx=12, pady=8)
            c.pack(side=tk.LEFT, padx=4)
            lbl(c, name.capitalize(), fg=BCOLORS[name], bg=BG3, size=9, bold=True).pack()
            v = tk.StringVar(value="—"); self.band_vars[name] = v
            lbl(c, "", fg=FG, bg=BG3, size=8, textvariable=v).pack()
        # Artifact card
        ac = tk.Frame(self.cards_frame, bg=BG3, padx=12, pady=8)
        ac.pack(side=tk.LEFT, padx=4)
        lbl(ac, "Artifacts", fg=WARN, bg=BG3, size=9, bold=True).pack()
        self.art_var = tk.StringVar(value="—")
        lbl(ac, "", fg=FG, bg=BG3, size=8, textvariable=self.art_var).pack()

        btn(self, "Refresh", self.refresh, color=ACC2).pack(anchor="e", padx=14, pady=4)

        if not MPL_OK:
            lbl(self, "pip install matplotlib  to enable plots", fg=WARN, bg=BG2).pack()
            return
        self.fig, self.axes = dark_fig(10, 5, 2, 1)
        self.ax_sig = self.axes[0]; self.ax_bp = self.axes[1]
        self.ax_sig.set_title("Raw Signal (µV)")
        self.ax_bp.set_title("Band Power Over Time")
        pf = tk.Frame(self, bg=BG2); pf.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)
        self.canvas = embed_fig(self.fig, pf)

    def refresh(self):
        d = get_active()
        if not d or d.raw is None:
            self.info_var.set("No session loaded."); return
        s = d.summary()
        self.info_var.set(
            f"File: {s['file']}  |  Start: {s['session_start']}  |  "
            f"Duration: {s['duration_s']}s  |  Samples: {s['total_samples']:,}  |  {s['sample_rate_hz']} Hz")
        for name in BANDS:
            self.band_vars[name].set(
                f"μ={np.mean(d.band[name]):.0f}\nσ={np.std(d.band[name]):.0f}")
        self.art_var.set(f"{s['artifact_pct']:.1f}%")
        if not MPL_OK: return
        self.ax_sig.cla(); self.ax_bp.cla()
        step = max(1, d.n // 5000)
        self.ax_sig.plot(d.t[::step], d.raw[::step], color=ACC, lw=0.5, alpha=0.85)
        self.ax_sig.set_xlabel("Time (s)"); self.ax_sig.set_ylabel("µV")
        self.ax_sig.set_title("Raw Signal (µV)")
        for n in BANDS:
            self.ax_bp.plot(d.bt, d.band_arr(n), color=BCOLORS[n], lw=0.9, label=n.capitalize())
        self.ax_bp.set_xlabel("Time (s)"); self.ax_bp.set_ylabel("Power (µV²/Hz)")
        self.ax_bp.set_title("Band Power Over Time")
        self.ax_bp.legend(fontsize=7, framealpha=0.2, loc="upper right")
        for ax in self.axes:
            for sp in ax.spines.values(): sp.set_edgecolor(BG4)
        self.canvas.draw()

# ═══════════════════════════════════════════════════════════════════════════
# Tab: Clean
# ═══════════════════════════════════════════════════════════════════════════
class CleanTab(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG2)
        self._build()

    def _build(self):
        ctrl = tk.LabelFrame(self, text="  Cleaning Controls  ", bg=BG2, fg=ACC,
                             font=("Segoe UI", 9, "bold"), padx=4, pady=4)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        # Amplitude threshold
        r0 = tk.Frame(ctrl, bg=BG2); r0.pack(fill=tk.X, pady=2)
        self.use_amp = tk.BooleanVar(value=True)
        make_check(r0, "Amplitude Threshold (µV):", self.use_amp, FG2).pack(side=tk.LEFT)
        self.amp_var = tk.DoubleVar(value=150.0)
        tk.Scale(r0, from_=10, to=1000, resolution=5, variable=self.amp_var,
                 orient=tk.HORIZONTAL, bg=BG2, fg=FG, troughcolor=BG3,
                 highlightthickness=0, length=200, showvalue=True,
                 activebackground=ACC).pack(side=tk.LEFT, padx=6)
        lbl(r0, "Rejects samples with |µV| above this value", fg=FG3, bg=BG2, size=8).pack(side=tk.LEFT, padx=6)

        # IQR
        r1 = tk.Frame(ctrl, bg=BG2); r1.pack(fill=tk.X, pady=2)
        self.use_iqr = tk.BooleanVar(value=False)
        make_check(r1, "IQR Outlier Multiplier:", self.use_iqr, FG2).pack(side=tk.LEFT)
        self.iqr_var = tk.DoubleVar(value=3.0)
        tk.Scale(r1, from_=1.0, to=10.0, resolution=0.5, variable=self.iqr_var,
                 orient=tk.HORIZONTAL, bg=BG2, fg=FG, troughcolor=BG3,
                 highlightthickness=0, length=200, showvalue=True,
                 activebackground=ACC).pack(side=tk.LEFT, padx=6)
        lbl(r1, "Rejects samples beyond Q1/Q3 ± mult×IQR", fg=FG3, bg=BG2, size=8).pack(side=tk.LEFT, padx=6)

        # Delta band artifact
        r2 = tk.Frame(ctrl, bg=BG2); r2.pack(fill=tk.X, pady=2)
        self.use_delta = tk.BooleanVar(value=True)
        make_check(r2, "Delta Artifact Threshold ×median:", self.use_delta, BCOLORS["delta"]).pack(side=tk.LEFT)
        self.delta_mult = tk.DoubleVar(value=3.0)
        tk.Scale(r2, from_=1.0, to=15.0, resolution=0.5, variable=self.delta_mult,
                 orient=tk.HORIZONTAL, bg=BG2, fg=FG, troughcolor=BG3,
                 highlightthickness=0, length=200, showvalue=True,
                 activebackground=BCOLORS["delta"]).pack(side=tk.LEFT, padx=6)
        lbl(r2, "Rejects entire 500ms batch if δ power > mult×median δ", fg=FG3, bg=BG2, size=8).pack(side=tk.LEFT, padx=6)

        # Beta band artifact
        r3 = tk.Frame(ctrl, bg=BG2); r3.pack(fill=tk.X, pady=2)
        self.use_beta = tk.BooleanVar(value=True)
        make_check(r3, "Beta Artifact Threshold ×median:", self.use_beta, BCOLORS["beta"]).pack(side=tk.LEFT)
        self.beta_mult = tk.DoubleVar(value=3.0)
        tk.Scale(r3, from_=1.0, to=15.0, resolution=0.5, variable=self.beta_mult,
                 orient=tk.HORIZONTAL, bg=BG2, fg=FG, troughcolor=BG3,
                 highlightthickness=0, length=200, showvalue=True,
                 activebackground=BCOLORS["beta"]).pack(side=tk.LEFT, padx=6)
        lbl(r3, "Rejects entire 500ms batch if β power > mult×median β (EMG / powerline)", fg=FG3, bg=BG2, size=8).pack(side=tk.LEFT, padx=6)

        # Buttons + status
        rb = tk.Frame(ctrl, bg=BG2); rb.pack(fill=tk.X, pady=6)
        btn(rb, "Apply Cleaning", self.apply, color=OK).pack(side=tk.LEFT, padx=4)
        btn(rb, "Reset to Raw",   self.reset, color=ERR).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="")
        lbl(rb, "", fg=OK, bg=BG2, size=9, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        if not MPL_OK:
            lbl(self, "pip install matplotlib  to enable plots", fg=WARN, bg=BG2).pack(); return
        self.fig, self.axes = dark_fig(10, 5, 2, 1)
        self.ax_b = self.axes[0]; self.ax_a = self.axes[1]
        self.ax_b.set_title("Before Cleaning"); self.ax_a.set_title("After Cleaning")
        pf = tk.Frame(self, bg=BG2); pf.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self.canvas = embed_fig(self.fig, pf)

    def apply(self):
        d = get_active()
        if not d or d.raw is None:
            messagebox.showwarning("No Data", "Load a session first."); return
        n = d.n; mask = np.zeros(n, dtype=bool); log = []

        if self.use_amp.get():
            thr = self.amp_var.get()
            m = Proc.amp_mask(d.raw, thr)
            mask |= m[:n]; log.append(f"Amplitude >{thr}µV")

        if self.use_iqr.get():
            mult = self.iqr_var.get()
            mask |= Proc.iqr_mask(d.raw, mult); log.append(f"IQR ×{mult}")

        spb = 128
        if self.use_delta.get():
            m = Proc.band_mask(d.band_arr("delta"), self.delta_mult.get(), spb)[:n]
            mask[:len(m)] |= m; log.append(f"δ artifact ×{self.delta_mult.get()}×med")

        if self.use_beta.get():
            m = Proc.band_mask(d.band_arr("beta"), self.beta_mult.get(), spb)[:n]
            mask[:len(m)] |= m; log.append(f"β artifact ×{self.beta_mult.get()}×med")

        d.art_mask = mask
        d.cleaned  = Proc.interp(d.raw, mask)
        d.preproc  = d.cleaned.copy()
        d.clean_steps = log
        pct = mask.mean() * 100
        self.status_var.set(f"✓  {pct:.1f}% artifacts removed  |  {'; '.join(log)}")

        if not MPL_OK: return
        step = max(1, n // 5000)
        t = d.t
        self.ax_b.cla(); self.ax_a.cla()
        self.ax_b.plot(t[::step], d.raw[::step], color=ACC, lw=0.4, alpha=0.8)
        art_idx = np.where(mask)[0][::max(1, mask.sum()//2000)]
        if len(art_idx):
            self.ax_b.scatter(t[art_idx], d.raw[art_idx], color=ERR, s=1.5, alpha=0.5,
                              zorder=3, label=f"artifact ({pct:.1f}%)")
            self.ax_b.legend(fontsize=7, framealpha=0.3)
        self.ax_b.set_title(f"Before — {pct:.1f}% flagged as artifact")
        self.ax_b.set_xlabel("Time (s)"); self.ax_b.set_ylabel("µV")
        self.ax_a.plot(t[::step], d.cleaned[::step], color=OK, lw=0.4, alpha=0.9)
        self.ax_a.set_title("After Cleaning (artifacts interpolated)")
        self.ax_a.set_xlabel("Time (s)"); self.ax_a.set_ylabel("µV")
        for ax in self.axes:
            for sp in ax.spines.values(): sp.set_edgecolor(BG4)
        self.canvas.draw()

    def reset(self):
        d = get_active()
        if not d: return
        d.art_mask = np.zeros(d.n, dtype=bool)
        d.cleaned = d.raw.copy(); d.preproc = d.raw.copy()
        d.clean_steps.clear()
        self.status_var.set("Reset — using raw signal")
        if MPL_OK:
            for ax in self.axes: ax.cla()
            self.canvas.draw()

# ═══════════════════════════════════════════════════════════════════════════
# Tab: Preprocess
# ═══════════════════════════════════════════════════════════════════════════
class PreprocessTab(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG2)
        self._build()

    def _build(self):
        ctrl = tk.LabelFrame(self, text="  Preprocessing Controls  ", bg=BG2, fg=ACC,
                             font=("Segoe UI", 9, "bold"), padx=4, pady=4)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        # Bandpass
        r0 = tk.Frame(ctrl, bg=BG2); r0.pack(fill=tk.X, pady=3)
        self.use_bp = tk.BooleanVar(value=True)
        make_check(r0, "Bandpass Filter:", self.use_bp, FG2).pack(side=tk.LEFT)
        lbl(r0, "Lo Hz:", fg=FG2, bg=BG2).pack(side=tk.LEFT, padx=(10,2))
        self.bp_lo = tk.DoubleVar(value=0.5)
        entry(r0, textvariable=self.bp_lo, width=6).pack(side=tk.LEFT)
        lbl(r0, "Hi Hz:", fg=FG2, bg=BG2).pack(side=tk.LEFT, padx=(8,2))
        self.bp_hi = tk.DoubleVar(value=45.0)
        entry(r0, textvariable=self.bp_hi, width=6).pack(side=tk.LEFT)
        lbl(r0, "Order:", fg=FG2, bg=BG2).pack(side=tk.LEFT, padx=(8,2))
        self.bp_order = tk.IntVar(value=4)
        entry(r0, textvariable=self.bp_order, width=3).pack(side=tk.LEFT)
        if not SCIPY_OK:
            lbl(r0, "  (scipy not installed — pip install scipy)", fg=WARN, bg=BG2, size=8).pack(side=tk.LEFT)

        # Notch
        r1 = tk.Frame(ctrl, bg=BG2); r1.pack(fill=tk.X, pady=3)
        self.use_notch = tk.BooleanVar(value=True)
        make_check(r1, "Notch Filter:", self.use_notch, FG2).pack(side=tk.LEFT)
        self.notch_freq = tk.DoubleVar(value=50.0)
        tk.OptionMenu(r1, self.notch_freq, 50.0, 60.0).pack(side=tk.LEFT, padx=6)
        lbl(r1, "Hz  (50 Hz for India powerline)", fg=FG3, bg=BG2, size=8).pack(side=tk.LEFT)

        # Detrend
        r2 = tk.Frame(ctrl, bg=BG2); r2.pack(fill=tk.X, pady=3)
        self.use_detrend = tk.BooleanVar(value=True)
        make_check(r2, "Detrend  (remove linear drift / DC offset)", self.use_detrend, FG2).pack(side=tk.LEFT)

        # Normalization
        r3 = tk.Frame(ctrl, bg=BG2); r3.pack(fill=tk.X, pady=3)
        lbl(r3, "Normalization:", fg=FG2, bg=BG2).pack(side=tk.LEFT)
        self.norm_var = tk.StringVar(value="none")
        for val, label in [("none","None"), ("zscore","Z-Score"), ("minmax","Min-Max [0,1]")]:
            tk.Radiobutton(r3, text=label, variable=self.norm_var, value=val,
                           bg=BG2, fg=FG2, selectcolor=BG3, activebackground=BG2,
                           font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)

        rb = tk.Frame(ctrl, bg=BG2); rb.pack(fill=tk.X, pady=6)
        btn(rb, "Apply Preprocessing", self.apply, color=OK).pack(side=tk.LEFT, padx=4)
        btn(rb, "Reset",               self.reset, color=ERR).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="")
        lbl(rb, "", fg=OK, bg=BG2, size=9, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        if not MPL_OK:
            lbl(self, "pip install matplotlib  to enable plots", fg=WARN, bg=BG2).pack(); return
        self.fig, self.axes = dark_fig(10, 5, 2, 1)
        self.ax_b = self.axes[0]; self.ax_a = self.axes[1]
        pf = tk.Frame(self, bg=BG2); pf.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self.canvas = embed_fig(self.fig, pf)

    def apply(self):
        d = get_active()
        if not d or d.cleaned is None:
            messagebox.showwarning("No Data", "Run Clean step first."); return
        sig = d.cleaned.copy(); log = []

        if self.use_notch.get():
            f = self.notch_freq.get()
            sig = Proc.notch(sig, f, d.fs); log.append(f"Notch {f:.0f}Hz")

        if self.use_bp.get():
            lo, hi = self.bp_lo.get(), self.bp_hi.get()
            sig = Proc.bandpass(sig, lo, hi, d.fs, int(self.bp_order.get()))
            log.append(f"Bandpass {lo}-{hi}Hz order {self.bp_order.get()}")

        if self.use_detrend.get():
            sig = Proc.detrend(sig); log.append("Detrend")

        mode = self.norm_var.get()
        if mode == "zscore":   sig = Proc.zscore(sig);  log.append("Z-score")
        elif mode == "minmax": sig = Proc.minmax(sig);  log.append("Min-max")

        d.preproc = sig; d.preproc_steps = log
        self.status_var.set(f"✓  {' → '.join(log)}")

        if not MPL_OK: return
        step = max(1, d.n // 5000); t = d.t
        self.ax_b.cla(); self.ax_a.cla()
        self.ax_b.plot(t[::step], d.cleaned[::step], color=ACC, lw=0.4)
        self.ax_b.set_title("Input (cleaned)"); self.ax_b.set_xlabel("Time (s)"); self.ax_b.set_ylabel("µV")
        self.ax_a.plot(t[::step], sig[::step], color=ACC2, lw=0.4)
        self.ax_a.set_title(f"Preprocessed: {' → '.join(log)}")
        self.ax_a.set_xlabel("Time (s)"); self.ax_a.set_ylabel("amplitude")
        for ax in self.axes:
            for sp in ax.spines.values(): sp.set_edgecolor(BG4)
        self.canvas.draw()

    def reset(self):
        d = get_active()
        if d:
            d.preproc = d.cleaned.copy(); d.preproc_steps.clear()
            self.status_var.set("Reset")

# ═══════════════════════════════════════════════════════════════════════════
# Tab: Annotate
# ═══════════════════════════════════════════════════════════════════════════
class AnnotateTab(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG2)
        self._build()

    def _build(self):
        lbl(self, "Event Annotation", fg=ACC, bg=BG2, size=12, bold=True).pack(anchor="w", padx=14, pady=(8,2))
        lbl(self, "Mark the times when questions/stimuli were presented. Click the timeline to place events.",
            fg=FG2, bg=BG2, size=9).pack(anchor="w", padx=14)

        # Timeline plot
        if MPL_OK:
            self.fig, self.ax = dark_fig(10, 2.8)
            self.ax.set_title("Band Power Timeline — left-click to set event time")
            pf = tk.Frame(self, bg=BG2); pf.pack(fill=tk.X, padx=14, pady=4)
            self.canvas = embed_fig(self.fig, pf)
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Entry form
        form = tk.LabelFrame(self, text="  Add / Edit Event  ", bg=BG2, fg=ACC,
                              font=("Segoe UI", 9, "bold"), padx=6, pady=4)
        form.pack(fill=tk.X, padx=14, pady=4)

        r0 = tk.Frame(form, bg=BG2); r0.pack(fill=tk.X, pady=3)
        lbl(r0, "Elapsed time (s):", fg=FG2, bg=BG2).pack(side=tk.LEFT)
        self.time_var = tk.StringVar(value="0.0")
        entry(r0, textvariable=self.time_var, width=10).pack(side=tk.LEFT, padx=6)
        btn(r0, "Auto-detect Dominant Band", self._auto_dom, color=ACC2).pack(side=tk.LEFT, padx=4)
        lbl(r0, "Dominant:", fg=FG2, bg=BG2).pack(side=tk.LEFT, padx=(12,4))
        self.dom_var = tk.StringVar(value="alpha")
        tk.OptionMenu(r0, self.dom_var, *BANDS).pack(side=tk.LEFT)

        r1 = tk.Frame(form, bg=BG2); r1.pack(fill=tk.X, pady=3)
        lbl(r1, "Question / Stimulus asked:", fg=FG2, bg=BG2).pack(side=tk.LEFT)
        self.q_var = tk.StringVar()
        entry(r1, textvariable=self.q_var, width=50).pack(side=tk.LEFT, padx=6)

        r2 = tk.Frame(form, bg=BG2); r2.pack(fill=tk.X, pady=3)
        lbl(r2, "Description / Notes:", fg=FG2, bg=BG2).pack(side=tk.LEFT)
        self.desc_var = tk.StringVar()
        entry(r2, textvariable=self.desc_var, width=60).pack(side=tk.LEFT, padx=6)

        rb = tk.Frame(form, bg=BG2); rb.pack(fill=tk.X, pady=5)
        btn(rb, "Add Event",       self._add,    color=OK).pack(side=tk.LEFT, padx=4)
        btn(rb, "Delete Selected", self._delete, color=ERR).pack(side=tk.LEFT, padx=4)
        btn(rb, "Refresh Timeline",self._refresh_plot, color=ACC2).pack(side=tk.LEFT, padx=4)

        # Table
        tf = tk.Frame(self, bg=BG2); tf.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)
        style = ttk.Style()
        style.configure("E.Treeview", background=BG3, foreground=FG,
                         fieldbackground=BG3, rowheight=24, font=("Segoe UI", 9))
        style.configure("E.Treeview.Heading", background=BG4, foreground=ACC,
                         font=("Segoe UI", 9, "bold"))
        style.map("E.Treeview", background=[("selected", BG4)], foreground=[("selected", ACC)])

        cols = ("elapsed_s","question","dominant","description")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings",
                                  height=7, style="E.Treeview")
        for col, w, head in [("elapsed_s",90,"Time (s)"),("question",260,"Question / Stimulus"),
                              ("dominant",110,"Dominant"),("description",370,"Description / Notes")]:
            self.tree.heading(col, text=head); self.tree.column(col, width=w)
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_click(self, event):
        if event.xdata is not None:
            self.time_var.set(str(round(event.xdata, 2)))
            self._auto_dom()

    def _auto_dom(self):
        d = get_active()
        if not d: return
        try: t = float(self.time_var.get()); self.dom_var.set(d.dominant_at(t))
        except Exception: pass

    def _add(self):
        d = get_active()
        if not d: return
        try: t = float(self.time_var.get())
        except ValueError: messagebox.showerror("Error","Invalid time."); return
        ev = {"elapsed_s": t, "question": self.q_var.get(),
              "dominant": self.dom_var.get(), "description": self.desc_var.get()}
        d.events.append(ev)
        self.tree.insert("", "end", values=(t, ev["question"], ev["dominant"], ev["description"]))
        self.q_var.set(""); self.desc_var.set("")
        self._refresh_plot()

    def _delete(self):
        d = get_active()
        if not d: return
        for item in self.tree.selection():
            idx = self.tree.index(item)
            if idx < len(d.events): d.events.pop(idx)
            self.tree.delete(item)
        self._refresh_plot()

    def _refresh_plot(self):
        d = get_active()
        if not d or not MPL_OK: return
        self.ax.cla()
        if d.raw is not None:
            for name in BANDS:
                self.ax.plot(d.bt, d.band_arr(name), color=BCOLORS[name], lw=0.8, label=name.capitalize())
            ymax = max(float(d.band_arr(n).max()) for n in BANDS) if d.n > 0 else 1.0
            for ev in d.events:
                self.ax.axvline(ev["elapsed_s"], color=WARN, lw=1.5, ls="--", alpha=0.85)
                q_short = ev["question"][:22] + "…" if len(ev["question"]) > 22 else ev["question"]
                self.ax.text(ev["elapsed_s"], ymax * 0.92, q_short,
                             rotation=90, fontsize=6, color=WARN, va="top", ha="right")
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Power")
        self.ax.set_title("Band Power Timeline — left-click to place event marker")
        self.ax.legend(fontsize=7, framealpha=0.2, loc="upper right")
        for sp in self.ax.spines.values(): sp.set_edgecolor(BG4)
        self.fig.canvas.draw()

    def load_events(self):
        d = get_active()
        if not d: return
        self.tree.delete(*self.tree.get_children())
        for ev in d.events:
            self.tree.insert("", "end", values=(ev["elapsed_s"],ev["question"],ev["dominant"],ev["description"]))
        self._refresh_plot()

# ═══════════════════════════════════════════════════════════════════════════
# Tab: Export
# ═══════════════════════════════════════════════════════════════════════════
class ExportTab(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG2)
        self._build()

    def _build(self):
        lbl(self, "Export Processed Data", fg=ACC, bg=BG2, size=12, bold=True).pack(anchor="w", padx=14, pady=(8,4))
        opts = tk.LabelFrame(self, text="  Export Options  ", bg=BG2, fg=ACC,
                              font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        opts.pack(fill=tk.X, padx=14, pady=6)

        self.ex_raw    = tk.BooleanVar(value=False)
        self.ex_clean  = tk.BooleanVar(value=True)
        self.ex_pre    = tk.BooleanVar(value=True)
        self.ex_events = tk.BooleanVar(value=True)
        self.ex_sum    = tk.BooleanVar(value=True)
        for var, text in [(self.ex_raw,   "Raw signal  → *_raw.npy"),
                          (self.ex_clean, "Cleaned signal  → *_cleaned.npy"),
                          (self.ex_pre,   "Preprocessed signal  → *_preprocessed.npy"),
                          (self.ex_events,"Event annotations  → *_events.csv"),
                          (self.ex_sum,   "Session summary  → *_summary.json")]:
            tk.Checkbutton(opts, text=text, variable=var, bg=BG2, fg=FG2,
                           selectcolor=BG3, activebackground=BG2,
                           font=("Segoe UI", 9)).pack(anchor="w", pady=2)

        btn(self, "Choose Folder & Export", self._export, color=OK).pack(anchor="w", padx=14, pady=8)

        self.log = scrolledtext.ScrolledText(self, height=14, bg=BG3, fg=FG2,
                                              font=("Consolas", 9), state="disabled", wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

    def _log(self, msg, color=FG2):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end"); self.log.config(state="disabled")

    def _export(self):
        d = get_active()
        if not d or d.raw is None:
            messagebox.showwarning("No Data","Load a session first."); return
        out_dir = filedialog.askdirectory(title="Select export folder",
                                          initialdir=r"c:\Users\User\Pictures\Neurosystem")
        if not out_dir: return
        base = Path(out_dir) / Path(d.filepath).stem
        if self.ex_raw.get():
            p = str(base)+"_raw.npy"; np.save(p, d.raw); self._log(f"✓ {p}")
        if self.ex_clean.get() and d.cleaned is not None:
            p = str(base)+"_cleaned.npy"; np.save(p, d.cleaned); self._log(f"✓ {p}")
        if self.ex_pre.get() and d.preproc is not None:
            p = str(base)+"_preprocessed.npy"; np.save(p, d.preproc); self._log(f"✓ {p}")
        if self.ex_events.get():
            p = str(base)+"_events.csv"
            with open(p,"w",newline="",encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["elapsed_s","question","dominant","description"])
                w.writeheader(); w.writerows(d.events)
            self._log(f"✓ {p}")
        if self.ex_sum.get():
            p = str(base)+"_summary.json"
            with open(p,"w",encoding="utf-8") as f: json.dump(d.summary(), f, indent=2)
            self._log(f"✓ {p}")
        self._log("Export complete.")
        messagebox.showinfo("Export",f"Files saved to:\n{out_dir}")

# ═══════════════════════════════════════════════════════════════════════════
# Tab: Claude AI
# ═══════════════════════════════════════════════════════════════════════════
class ClaudeTab(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG2)
        self._history = []
        self._build()

    def _build(self):
        lbl(self, "Claude AI — EEG Analysis Assistant", fg=ACC, bg=BG2, size=12, bold=True).pack(anchor="w", padx=14, pady=(8,2))
        lbl(self, "Current session context (band powers, artifact %, pipeline steps, events) is sent automatically with every message.",
            fg=FG2, bg=BG2, size=9).pack(anchor="w", padx=14)

        kr = tk.Frame(self, bg=BG2); kr.pack(fill=tk.X, padx=14, pady=6)
        lbl(kr, "Anthropic API Key:", fg=FG2, bg=BG2).pack(side=tk.LEFT)
        self.key_var = tk.StringVar(value=os.environ.get("ANTHROPIC_API_KEY",""))
        self.key_entry = tk.Entry(kr, textvariable=self.key_var, width=55, show="*",
                                   bg=BG3, fg=FG, insertbackground=FG,
                                   relief=tk.FLAT, font=("Consolas", 9))
        self.key_entry.pack(side=tk.LEFT, padx=8)
        self.show_key = tk.BooleanVar(value=False)
        def _toggle():
            self.key_entry.config(show="" if self.show_key.get() else "*")
        tk.Checkbutton(kr, text="Show", variable=self.show_key, command=_toggle,
                       bg=BG2, fg=FG2, selectcolor=BG3, activebackground=BG2).pack(side=tk.LEFT)

        if not ANTHROPIC_OK:
            lbl(self, "  anthropic not installed — run:  pip install anthropic",
                fg=ERR, bg=BG2, size=9).pack(anchor="w", padx=14, pady=2)

        # Chat display
        self.chat = scrolledtext.ScrolledText(self, height=22, bg=BG3, fg=FG,
                                               font=("Consolas", 9), state="disabled",
                                               wrap=tk.WORD)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)
        self.chat.tag_configure("user",   foreground=ACC,  font=("Consolas", 9, "bold"))
        self.chat.tag_configure("claude", foreground=OK,   font=("Consolas", 9))
        self.chat.tag_configure("sys",    foreground=FG2,  font=("Consolas", 8, "italic"))
        self.chat.tag_configure("err",    foreground=ERR)

        # Input row
        ir = tk.Frame(self, bg=BG2); ir.pack(fill=tk.X, padx=14, pady=6)
        self.inp_var = tk.StringVar()
        inp = tk.Entry(ir, textvariable=self.inp_var, bg=BG3, fg=FG, insertbackground=FG,
                       relief=tk.FLAT, font=("Segoe UI", 10))
        inp.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,8))
        inp.bind("<Return>", lambda _: self._send())
        btn(ir, "Send", self._send, color=ACC).pack(side=tk.LEFT)
        btn(ir, "Show Context", self._show_context, color=ACC2).pack(side=tk.LEFT, padx=4)
        btn(ir, "Clear Chat", self._clear, color=BG4, fg=FG2).pack(side=tk.LEFT, padx=4)

        self._write("Session loaded — type a question and press Send or Enter.\n"
                    "Example: 'What do the delta/beta spikes indicate? What thresholds should I use?'\n", "sys")

    def _write(self, text, tag="sys"):
        self.chat.config(state="normal")
        self.chat.insert("end", text + "\n", tag)
        self.chat.see("end"); self.chat.config(state="disabled")

    def _show_context(self):
        d = get_active()
        ctx = json.dumps(d.summary(), indent=2) if d else "No session loaded."
        self._write(f"--- Current context ---\n{ctx}\n---", "sys")

    def _clear(self):
        self.chat.config(state="normal"); self.chat.delete("1.0","end")
        self.chat.config(state="disabled"); self._history.clear()

    def _build_ctx(self):
        d = get_active()
        if not d: return "No EEG session loaded."
        s = d.summary()
        return (
            f"EEG session context:\n"
            f"  File: {s.get('file')}\n"
            f"  Start: {s.get('session_start')}  Duration: {s.get('duration_s')}s\n"
            f"  Samples: {s.get('total_samples'):,}  Sample rate: {s.get('sample_rate_hz')} Hz\n"
            f"  Raw signal µV range: {s.get('raw_uV')}\n"
            f"  Band power means (µV²/Hz): {s.get('band_power_means')}\n"
            f"  Artifact %: {s.get('artifact_pct')}\n"
            f"  Cleaning steps applied: {s.get('clean_steps')}\n"
            f"  Preprocessing steps applied: {s.get('preproc_steps')}\n"
            f"  Events annotated: {len(s.get('events', []))}\n"
            f"  Event details: {s.get('events')}\n"
            f"  Hardware: BioAmp EXG Pill + ESP32, single-channel Fp1-Fp2 differential (frontal EEG)\n"
        )

    def _send(self):
        if not ANTHROPIC_OK:
            self._write("anthropic not installed. Run: pip install anthropic", "err"); return
        key = self.key_var.get().strip()
        if not key:
            self._write("Enter your Anthropic API key above.", "err"); return
        msg = self.inp_var.get().strip()
        if not msg: return
        self.inp_var.set("")
        self._write(f"You: {msg}", "user")

        ctx = self._build_ctx()
        full = f"{ctx}\n\nUser question: {msg}"
        self._history.append({"role":"user","content": full})

        # Placeholder
        self.chat.config(state="normal")
        self.chat.insert("end", "Claude: thinking…\n", "sys")
        self.chat.see("end"); self.chat.config(state="disabled")

        root = self.winfo_toplevel()

        def worker():
            try:
                client = anthropic.Anthropic(api_key=key)
                resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    system=(
                        "You are an expert neuroscientist and EEG signal processing engineer. "
                        "You are embedded in a desktop EEG data pipeline tool. "
                        "The user is recording with a BioAmp EXG Pill + ESP32, single-channel "
                        "Fp1-Fp2 differential frontal EEG at 256 Hz. "
                        "Help them: (1) set cleaning thresholds to remove delta/beta false positives, "
                        "(2) choose correct bandpass/notch filter settings, "
                        "(3) interpret band power patterns relative to stimuli/questions, "
                        "(4) guide feature extraction and model training decisions. "
                        "Be concise, technical, and actionable. Use bullet points when listing recommendations."
                    ),
                    messages=self._history,
                )
                reply = resp.content[0].text
                self._history.append({"role":"assistant","content": reply})
                def _update():
                    self.chat.config(state="normal")
                    # Remove the "thinking…" line
                    self.chat.delete("end-2l linestart", "end-1c")
                    self.chat.insert("end", f"Claude: {reply}\n\n", "claude")
                    self.chat.see("end"); self.chat.config(state="disabled")
                root.after(0, _update)
            except Exception as ex:
                def _err():
                    self.chat.config(state="normal")
                    self.chat.delete("end-2l linestart","end-1c")
                    self.chat.insert("end", f"Error: {ex}\n", "err")
                    self.chat.config(state="disabled")
                root.after(0, _err)

        threading.Thread(target=worker, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════════
# Main application
# ═══════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neurosystem — EEG Training Studio")
        self.geometry("1300x820"); self.minsize(960, 640)
        self.configure(bg=BG)
        self._style(); self._layout(); self._check_deps()

    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                     padding=[14,7], font=("Segoe UI", 9))
        s.map("TNotebook.Tab",
              background=[("selected", BG2)], foreground=[("selected", ACC)])
        s.configure("TScrollbar", background=BG3, troughcolor=BG,
                     arrowcolor=FG2, borderwidth=0)
        s.configure("TSeparator", background=BG4)

    def _layout(self):
        # Top bar
        top = tk.Frame(self, bg=BG, height=52); top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)
        lbl(top, "  Neurosystem", fg=ACC, bg=BG, size=14, bold=True).pack(side=tk.LEFT, padx=4, pady=8)
        lbl(top, "EEG Training Studio  |  BioAmp EXG Pill + ESP32",
            fg=FG2, bg=BG, size=9).pack(side=tk.LEFT, padx=4)
        self.status_top = tk.StringVar(value="Ready")
        lbl(top, "", fg=OK, bg=BG, size=9, textvariable=self.status_top).pack(side=tk.RIGHT, padx=14)

        # Main container
        main = tk.Frame(self, bg=BG); main.pack(fill=tk.BOTH, expand=True)

        # ── Sidebar ──────────────────────────────────────────────────────
        sb = tk.Frame(main, bg=BG2, width=220); sb.pack(fill=tk.Y, side=tk.LEFT)
        sb.pack_propagate(False)

        lbl(sb, "Sessions", fg=ACC, bg=BG2, size=10, bold=True).pack(anchor="w", padx=10, pady=(14,4))
        self.lb = tk.Listbox(sb, bg=BG3, fg=FG, selectbackground=BG4, selectforeground=ACC,
                              height=7, borderwidth=0, highlightthickness=0, font=("Segoe UI", 8))
        self.lb.pack(fill=tk.X, padx=8, pady=2)
        self.lb.bind("<<ListboxSelect>>", self._on_select)

        btn(sb, "+ Load File",    self._load_file, color=ACC).pack(fill=tk.X, padx=8, pady=2)
        btn(sb, "Load Both Sessions", self._load_both, color=ACC2).pack(fill=tk.X, padx=8, pady=2)

        ttk.Separator(sb, orient="horizontal").pack(fill=tk.X, padx=8, pady=10)

        lbl(sb, "Pipeline Status", fg=ACC, bg=BG2, size=9, bold=True).pack(anchor="w", padx=10, pady=2)
        self.step_vars = {}
        for step, icon in [("loaded","○"),("cleaned","○"),("preprocessed","○"),
                            ("annotated","○"),("exported","○")]:
            fr = tk.Frame(sb, bg=BG2); fr.pack(fill=tk.X, padx=10, pady=1)
            icon_var = tk.StringVar(value=icon)
            self.step_vars[step] = icon_var
            lbl(fr, "", fg=FG3, bg=BG2, size=10, textvariable=icon_var).pack(side=tk.LEFT)
            lbl(fr, f"  {step.capitalize()}", fg=FG2, bg=BG2, size=9).pack(side=tk.LEFT)

        ttk.Separator(sb, orient="horizontal").pack(fill=tk.X, padx=8, pady=10)

        lbl(sb, "Info", fg=ACC, bg=BG2, size=9, bold=True).pack(anchor="w", padx=10, pady=2)
        self.info = scrolledtext.ScrolledText(sb, height=10, bg=BG3, fg=FG2,
                                               font=("Consolas", 8), state="disabled",
                                               wrap=tk.WORD, borderwidth=0)
        self.info.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,10))

        # ── Notebook ─────────────────────────────────────────────────────
        self.nb = ttk.Notebook(main); self.nb.pack(fill=tk.BOTH, expand=True)
        self.t_overview = OverviewTab(self.nb)
        self.t_clean    = CleanTab(self.nb)
        self.t_preproc  = PreprocessTab(self.nb)
        self.t_annotate = AnnotateTab(self.nb)
        self.t_export   = ExportTab(self.nb)
        self.t_claude   = ClaudeTab(self.nb)
        for tab, label in [(self.t_overview, "  Overview  "),
                           (self.t_clean,    "  Clean  "),
                           (self.t_preproc,  "  Preprocess  "),
                           (self.t_annotate, "  Annotate  "),
                           (self.t_export,   "  Export  "),
                           (self.t_claude,   "  Claude AI  ")]:
            self.nb.add(tab, text=label)

    def _check_deps(self):
        missing = []
        if not SCIPY_OK:     missing.append("scipy")
        if not MPL_OK:       missing.append("matplotlib")
        if not ANTHROPIC_OK: missing.append("anthropic")
        if missing:
            self._log(f"Missing packages detected.\nInstall with:\n  pip install {' '.join(missing)}\n")

    def _log(self, msg):
        self.info.config(state="normal")
        self.info.insert("end", msg + "\n")
        self.info.see("end"); self.info.config(state="disabled")

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Open EEG Session JSON",
            initialdir=r"c:\Users\User\Pictures\Neurosystem",
            filetypes=[("JSON files","*.json"),("All files","*.*")])
        if path: self._load_session(path)

    def _load_both(self):
        base = Path(r"c:\Users\User\Pictures\Neurosystem")
        for name in ["eeg_session_2026-04-14T22-25-48.json",
                     "eeg_session_2026-04-14T22-25-48 (1).json"]:
            p = base / name
            if p.exists(): self._load_session(str(p))
            else: self._log(f"Not found: {name}")

    def _load_session(self, path):
        label = Path(path).name
        if label in sessions:
            self._log(f"Already loaded: {label}"); return
        try:
            d = EEGData(); d.load(path); sessions[label] = d
            active_session["key"] = label
            self.lb.insert("end", label)
            self.lb.selection_clear(0,"end"); self.lb.selection_set("end")
            self._log(f"Loaded: {label}\n"
                      f"  Duration: {d.duration_s:.1f}s\n"
                      f"  Samples:  {d.n:,}\n"
                      f"  Batches:  {len(d.batches)}\n")
            self.step_vars["loaded"].set("●")
            self.status_top.set(f"Active: {label}")
            self.t_overview.refresh()
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))
            self._log(f"Error: {ex}")

    def _on_select(self, _):
        sel = self.lb.curselection()
        if not sel: return
        label = self.lb.get(sel[0]); active_session["key"] = label
        d = sessions.get(label)
        if d:
            self.status_top.set(f"Active: {label}")
            self._log(f"Switched to: {label}")
            self.t_overview.refresh()
            self.t_annotate.load_events()

if __name__ == "__main__":
    app = App()
    app.mainloop()
