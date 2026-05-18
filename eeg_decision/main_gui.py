#!/usr/bin/env python3
"""
Neurosystem — EEG Decision Detection Studio
3 Wings: Analysis & Testing | Subject Testing | Prepare Model
Run: python eeg_decision/main_gui.py  (from Neurosystem folder)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, queue, time, csv, json, math
from datetime import datetime
from pathlib import Path
import numpy as np

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    from eeg_decision.streaming.ws_reader import WSReader, WS_OK
except Exception:
    WS_OK = False

try:
    from eeg_decision.features.ssvep import ssvep_powers, full_spectrum
    from eeg_decision.features.band_power import feature_vector
    from eeg_decision.classifier.train import (train_from_epochs, train_from_csv,
                                               train_from_feature_arrays)
    from eeg_decision.classifier.predict import Predictor
    MODULES_OK = True
except Exception as e:
    MODULES_OK = False
    _MODULE_ERR = str(e)

from eeg_decision.config import *

try:
    from eeg_decision.data import mongo_store
    MONGO_STORE_OK = mongo_store.MONGO_OK
except Exception:
    mongo_store    = None   # type: ignore
    MONGO_STORE_OK = False

# ── Palette helpers ────────────────────────────────────────────────────────────
def lbl(p, text="", fg=FG, bg=BG2, size=9, bold=False, **kw):
    return tk.Label(p, text=text, fg=fg, bg=bg,
                    font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

def btn(p, text, cmd, color=ACC, fg_col=BG, **kw):
    return tk.Button(p, text=text, command=cmd, bg=color, fg=fg_col,
                     activebackground=color, activeforeground=fg_col,
                     relief=tk.FLAT, padx=10, pady=5,
                     font=("Segoe UI", 9, "bold"), cursor="hand2", **kw)

def frame(p, bg=BG2, **kw):
    return tk.Frame(p, bg=bg, **kw)

def dark_fig(w=9, h=3.5, rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h),
                             facecolor=BG2, constrained_layout=True)
    ax_list = axes.flat if hasattr(axes, "flat") else [axes]
    for ax in ax_list:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG2)
        for sp in ax.spines.values(): sp.set_edgecolor(BG4)
        ax.xaxis.label.set_color(FG2); ax.yaxis.label.set_color(FG2)
        ax.title.set_color(FG)
    return fig, axes

# ── Shared live data state ─────────────────────────────────────────────────────
class LiveState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.waveform    = np.zeros(WAVEFORM_PTS, dtype=np.float32)   # Ch1 rolling
        self.waveform2   = np.zeros(WAVEFORM_PTS, dtype=np.float32)   # Ch2 rolling
        # Ch1
        self.band_rel    = {n: 0.0 for n in BANDS}
        self.quality     = "disconnected"
        self.ab_ratio    = 0.0
        self.ta_ratio    = 0.0
        self.filt_rms    = 0.0
        # Ch2
        self.band_rel2   = {n: 0.0 for n in BANDS}
        self.quality2    = "disconnected"
        self.ab_ratio2   = 0.0
        self.ta_ratio2   = 0.0
        self.filt_rms2   = 0.0
        # shared
        self.epoch_no    = 0
        self.ssvep_p     = {f: 0.0 for f in SSVEP_FREQS}
        self.spectrum_f  = np.array([])
        self.spectrum_p  = np.array([])
        self.last_epoch  = None   # raw JSON dict

    def ingest(self, msg: dict):
        with self.lock:
            self.last_epoch  = msg
            self.epoch_no    = msg.get("epoch", self.epoch_no)
            # Ch1
            self.quality     = msg.get("quality",   "disconnected")
            self.ab_ratio    = msg.get("ab_ratio",   0.0)
            self.ta_ratio    = msg.get("ta_ratio",   0.0)
            self.filt_rms    = msg.get("filt_rms",   0.0)
            rel = msg.get("rel", {})
            for k, n in [("d","delta"),("t","theta"),("a","alpha"),("b","beta"),("g","gamma")]:
                self.band_rel[n] = float(rel.get(k, 0.0))
            # Ch2
            self.quality2    = msg.get("quality2",  "disconnected")
            self.ab_ratio2   = msg.get("ab_ratio2",  0.0)
            self.ta_ratio2   = msg.get("ta_ratio2",  0.0)
            self.filt_rms2   = msg.get("filt_rms2",  0.0)
            rel2 = msg.get("rel2", {})
            for k, n in [("d","delta"),("t","theta"),("a","alpha"),("b","beta"),("g","gamma")]:
                self.band_rel2[n] = float(rel2.get(k, 0.0))
            # roll waveforms
            ch = np.array(msg.get("ch", []), dtype=np.float32) / 10.0
            if len(ch):
                self.waveform = np.roll(self.waveform, -len(ch))
                self.waveform[-len(ch):] = ch
            ch2 = np.array(msg.get("ch2", []), dtype=np.float32) / 10.0
            if len(ch2):
                self.waveform2 = np.roll(self.waveform2, -len(ch2))
                self.waveform2[-len(ch2):] = ch2
            # SSVEP from Ch1 (frontal — best for attention detection)
            if len(ch) == N_SAMPLES:
                self.ssvep_p = ssvep_powers(ch) if MODULES_OK else {}
                sf, sp = full_spectrum(ch) if MODULES_OK else (np.array([]), np.array([]))
                self.spectrum_f = sf; self.spectrum_p = sp

LIVE = LiveState()
PREDICTOR = Predictor() if MODULES_OK else None

# ═══════════════════════════════════════════════════════════════════════════════
# Wing 1 — Analysis & Testing
# ═══════════════════════════════════════════════════════════════════════════════
class WingAnalysis(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG)
        self._csv_rows      = []        # live-recorded rows
        self._rt_epochs     = []        # real-time training epochs
        self._rt_labels     = []        # matching labels
        self._rt_cur_label  = tk.IntVar(value=0)
        self._rt_recording  = False
        self._build()

    def _build(self):
        # ── top bar ──
        top = frame(self, bg=BG); top.pack(fill=tk.X, padx=12, pady=(10, 4))
        lbl(top, "Analysis & Testing", fg=ACC, bg=BG, size=13, bold=True).pack(side=tk.LEFT)
        self._q_badge2 = lbl(top, "Ch2: ○  disconnected", fg=FG3, bg=BG, size=8)
        self._q_badge2.pack(side=tk.RIGHT, padx=4)
        self._q_badge = lbl(top, "  Ch1: ○  disconnected", fg=FG3, bg=BG, size=8)
        self._q_badge.pack(side=tk.RIGHT, padx=4)

        # ── main split ──
        mid = frame(self, bg=BG); mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left: waveform + SSVEP spectrum
        left = frame(mid, bg=BG); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_waveform(left)

        # Right: band bars + metrics
        right = frame(mid, bg=BG2, width=230); right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)
        self._build_band_panel(right)

        # ── bottom action bar ──
        self._build_actions()

    def _build_waveform(self, parent):
        if not MPL_OK:
            lbl(parent, "pip install matplotlib to enable waveform", fg=WARN, bg=BG).pack()
            return
        self._fig, axes = dark_fig(9, 4.2, 2, 1)
        self._ax1 = axes[0]; self._ax2 = axes[1]
        self._ax1.set_title(f"Ch1 — {CH1_LABEL}  (µV)", fontsize=9)
        self._ax2.set_title(f"Ch2 — {CH2_LABEL}  (µV)", fontsize=9)
        t = np.linspace(-WAVEFORM_SECS, 0, WAVEFORM_PTS)
        self._line1, = self._ax1.plot(t, np.zeros(WAVEFORM_PTS), color=ACC, lw=0.8)
        self._line2, = self._ax2.plot(t, np.zeros(WAVEFORM_PTS), color=ACC2, lw=0.8)
        self._ax1.set_ylim(-120, 120); self._ax2.set_ylim(-120, 120)
        self._ax1.set_xlabel("Time (s)"); self._ax2.set_xlabel("Time (s)")
        canvas = FigureCanvasTkAgg(self._fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._mpl_canvas = canvas

    def _build_band_panel(self, parent):
        lbl(parent, "Band Power", fg=ACC, bg=BG2, size=9, bold=True).pack(
            anchor="w", padx=10, pady=(12, 4))
        self._band_bars  = {}
        self._band_vals  = {}
        for name in BANDS:
            color = BAND_COLORS[name]
            fr = frame(parent, bg=BG2); fr.pack(fill=tk.X, padx=10, pady=3)
            lbl(fr, name.capitalize(), fg=color, bg=BG2, size=8, bold=True).pack(anchor="w")
            bar_bg = frame(fr, bg=BG3, height=7); bar_bg.pack(fill=tk.X)
            bar_bg.pack_propagate(False)
            bar = frame(bar_bg, bg=color, height=7, width=2); bar.place(x=0, y=0, relheight=1)
            v = tk.StringVar(value="0.0 %")
            lbl(fr, "", fg=FG2, bg=BG2, size=8, textvariable=v).pack(anchor="e")
            self._band_bars[name]  = (bar, bar_bg)
            self._band_vals[name]  = v

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)
        lbl(parent, "Ch1 Metrics", fg=ACC, bg=BG2, size=9, bold=True).pack(anchor="w", padx=10)
        self._met_vars = {}
        for key, label in [("ab","α/β ratio"),("ta","θ/α ratio"),("rms","Filt RMS µV"),
                            ("ssvep8","SSVEP 8Hz"),("ssvep10","SSVEP 10Hz"),("ssvep12","SSVEP 12Hz")]:
            fr = frame(parent, bg=BG2); fr.pack(fill=tk.X, padx=10, pady=2)
            lbl(fr, label + ":", fg=FG2, bg=BG2, size=8).pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            lbl(fr, "", fg=FG, bg=BG2, size=8, textvariable=v).pack(side=tk.RIGHT)
            self._met_vars[key] = v

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)
        lbl(parent, "Ch2 Metrics", fg=ACC2, bg=BG2, size=9, bold=True).pack(anchor="w", padx=10)
        self._met2_vars = {}
        for key, label in [("ab2","α/β ratio"),("ta2","θ/α ratio"),("rms2","Filt RMS µV")]:
            fr = frame(parent, bg=BG2); fr.pack(fill=tk.X, padx=10, pady=2)
            lbl(fr, label + ":", fg=FG2, bg=BG2, size=8).pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            lbl(fr, "", fg=ACC2, bg=BG2, size=8, textvariable=v).pack(side=tk.RIGHT)
            self._met2_vars[key] = v

    def _build_actions(self):
        ab = frame(self, bg=BG3); ab.pack(fill=tk.X, padx=12, pady=6)

        # Download CSV
        btn(ab, "Download CSV", self._download_csv, color=BG4, fg_col=FG).pack(side=tk.LEFT, padx=4, pady=6)

        # Train from CSV
        btn(ab, "Train from CSV", self._train_csv, color=ACC2, fg_col=BG).pack(side=tk.LEFT, padx=4, pady=6)

        ttk.Separator(ab, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Real-time training
        lbl(ab, "Real-time train — label:", fg=FG2, bg=BG3, size=8).pack(side=tk.LEFT, padx=(4,2), pady=6)
        for i, name in enumerate(CLASS_NAMES):
            color = SSVEP_COLORS[i] if i < len(SSVEP_COLORS) else FG2
            tk.Radiobutton(ab, text=name, variable=self._rt_cur_label, value=i,
                           bg=BG3, fg=color, selectcolor=BG4, activebackground=BG3,
                           font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=3, pady=6)

        self._rt_btn = btn(ab, "Start Recording", self._toggle_rt, color=OK, fg_col=BG)
        self._rt_btn.pack(side=tk.LEFT, padx=6, pady=6)
        self._rt_status = lbl(ab, "0 epochs collected", fg=FG2, bg=BG3, size=8)
        self._rt_status.pack(side=tk.LEFT, padx=4)

        self._train_btn = btn(ab, "Train Now", self._train_rt, color=WARN, fg_col=BG)
        self._train_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self._train_log = lbl(ab, "", fg=OK, bg=BG3, size=8)
        self._train_log.pack(side=tk.LEFT, padx=8)

    # ── update called by App every 100 ms ─────────────────────────────────────

    def update_display(self):
        with LIVE.lock:
            q    = LIVE.quality;    q2   = LIVE.quality2
            rel  = dict(LIVE.band_rel);   rel2 = dict(LIVE.band_rel2)
            ab   = LIVE.ab_ratio;   ta   = LIVE.ta_ratio;  rms  = LIVE.filt_rms
            ab2  = LIVE.ab_ratio2;  ta2  = LIVE.ta_ratio2; rms2 = LIVE.filt_rms2
            sp   = dict(LIVE.ssvep_p)
            w1   = LIVE.waveform.copy()
            w2   = LIVE.waveform2.copy()
            ep   = LIVE.last_epoch

        # quality badges
        cfg = {"good":(OK,"●"), "floating":(WARN,"◐"),
               "flat":(WARN,"○"), "disconnected":(ERR,"○")}
        c1, i1 = cfg.get(q,  (FG3,"?"))
        c2, i2 = cfg.get(q2, (FG3,"?"))
        self._q_badge.config( text=f"  Ch1: {i1}  {q}",  fg=c1)
        self._q_badge2.config(text=f"Ch2: {i2}  {q2}", fg=c2)

        # band bars (Ch1)
        for name in BANDS:
            pct = rel.get(name, 0.0) * 100
            v, (bar, bg) = self._band_vals[name], self._band_bars[name]
            v.set(f"{pct:.1f} %")
            bg.update_idletasks()
            w = bg.winfo_width()
            bar.place(x=0, y=0, relheight=1, width=max(2, int(w * pct / 100)))

        # Ch1 metrics
        self._met_vars["ab"].set(f"{ab:.3f}")
        self._met_vars["ta"].set(f"{ta:.3f}")
        self._met_vars["rms"].set(f"{rms:.1f}")
        for freq, key in zip(SSVEP_FREQS, ["ssvep8","ssvep10","ssvep12"]):
            self._met_vars[key].set(f"{sp.get(freq,0):.0f}")

        # Ch2 metrics
        self._met2_vars["ab2"].set(f"{ab2:.3f}")
        self._met2_vars["ta2"].set(f"{ta2:.3f}")
        self._met2_vars["rms2"].set(f"{rms2:.1f}")

        # waveform
        if MPL_OK:
            self._line1.set_ydata(w1)
            self._line2.set_ydata(w2)
            for ax, sig in [(self._ax1, w1), (self._ax2, w2)]:
                vmax = max(30.0, float(np.abs(sig).max()) * 1.15)
                ax.set_ylim(-vmax, vmax)
            self._mpl_canvas.draw_idle()

        # real-time recording
        if self._rt_recording and ep is not None:
            self._rt_epochs.append(ep)
            self._rt_labels.append(self._rt_cur_label.get())
            cnt = len(self._rt_epochs)
            self._rt_status.config(
                text=f"{cnt} epochs  ({', '.join(f'{CLASS_NAMES[i]}:{self._rt_labels.count(i)}' for i in range(N_CLASSES))})")

        # CSV logging — both channels
        if ep is not None:
            r1 = ep.get("rel",  {})
            r2 = ep.get("rel2", {})
            self._csv_rows.append({
                "epoch":    ep.get("epoch",""),
                "ms":       ep.get("ts",""),
                # Ch1
                "quality":  ep.get("quality",""),
                "filt_rms": ep.get("filt_rms",""),
                "D_rel":  r1.get("d",""), "T_rel":  r1.get("t",""),
                "A_rel":  r1.get("a",""), "B_rel":  r1.get("b",""),
                "G_rel":  r1.get("g",""),
                "A_B_ratio": ep.get("ab_ratio",""),
                "T_A_ratio": ep.get("ta_ratio",""),
                # Ch2
                "quality2":  ep.get("quality2",""),
                "filt_rms2": ep.get("filt_rms2",""),
                "D2_rel": r2.get("d",""), "T2_rel": r2.get("t",""),
                "A2_rel": r2.get("a",""), "B2_rel": r2.get("b",""),
                "G2_rel": r2.get("g",""),
                "A_B_ratio2": ep.get("ab_ratio2",""),
                "T_A_ratio2": ep.get("ta_ratio2",""),
            })

    # ── actions ───────────────────────────────────────────────────────────────

    def _download_csv(self):
        if not self._csv_rows:
            messagebox.showinfo("No data", "No live epochs recorded yet."); return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).parent / "data" / f"eeg_live_{ts}.csv"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self._csv_rows[0].keys()))
            w.writeheader(); w.writerows(self._csv_rows)
        messagebox.showinfo("Saved", f"CSV saved:\n{out}")

    def _train_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file", filetypes=[("CSV","*.csv"),("All","*.*")],
            initialdir=str(Path(__file__).parent.parent / "BioAmp_EXG_EEG_ESP32"))
        if not path: return
        self._train_log.config(text="Training…", fg=WARN)
        def _run():
            res = train_from_csv(path)
            def _done():
                color = OK if res["ok"] else ERR
                self._train_log.config(text=res["msg"], fg=color)
                if res["ok"] and PREDICTOR:
                    PREDICTOR.set_model(res["model"], res["feature_dim"],
                                        res.get("label_map"))
            self.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

    def _toggle_rt(self):
        self._rt_recording = not self._rt_recording
        if self._rt_recording:
            self._rt_btn.config(text="Stop Recording", bg=ERR)
        else:
            self._rt_btn.config(text="Start Recording", bg=OK)

    def _train_rt(self):
        if len(self._rt_epochs) < 4:
            messagebox.showinfo("Not enough data",
                                "Need at least 4 labeled epochs. Keep recording.")
            return
        self._train_log.config(text="Training…", fg=WARN)
        eps, lbls = list(self._rt_epochs), list(self._rt_labels)
        def _run():
            res = train_from_epochs(eps, lbls)
            def _done():
                color = OK if res["ok"] else ERR
                self._train_log.config(text=res["msg"], fg=color)
                if res["ok"] and PREDICTOR:
                    PREDICTOR.set_model(res["model"], res["feature_dim"])
            self.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Wing 2 — Subject Testing (SSVEP color spectrum)
# ═══════════════════════════════════════════════════════════════════════════════
class WingSubject(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG)
        self._frame_no = 0
        self._stim_on  = False
        self._build()
        self._tick()

    def _build(self):
        lbl(self, "Subject Testing — SSVEP Color Spectrum", fg=ACC, bg=BG,
            size=13, bold=True).pack(anchor="w", padx=14, pady=(10, 4))
        lbl(self, "Focus on one colour. EEG will detect which one you're looking at.",
            fg=FG2, bg=BG, size=9).pack(anchor="w", padx=14)

        # ── stimulus canvas ───────────────────────────────────────────────────
        stim_frame = frame(self, bg=BG); stim_frame.pack(fill=tk.X, padx=14, pady=8)

        self._stim_btn = btn(stim_frame, "Start Stimulus", self._toggle_stim,
                              color=OK, fg_col=BG)
        self._stim_btn.pack(side=tk.LEFT, pady=4)
        lbl(stim_frame, "  Red = 8 Hz    Green = 10 Hz    Violet = 12 Hz",
            fg=FG2, bg=BG, size=8).pack(side=tk.LEFT, padx=12)

        self._canvas = tk.Canvas(self, bg="#0a0a0a", height=240,
                                  highlightthickness=0)
        self._canvas.pack(fill=tk.X, padx=14, pady=4)
        self._canvas.bind("<Configure>", self._on_resize)
        self._rects = []
        self._freq_labels = []
        self._build_rects()

        # ── FFT spectrum bar chart ─────────────────────────────────────────────
        spec_lf = tk.LabelFrame(self, text="  Live SSVEP Spectrum (1–30 Hz)  ",
                                 bg=BG2, fg=ACC, font=("Segoe UI", 9, "bold"),
                                 padx=6, pady=6)
        spec_lf.pack(fill=tk.X, padx=14, pady=4)

        if MPL_OK:
            self._spec_fig, self._spec_ax = dark_fig(9, 2.0)
            self._spec_ax.set_xlabel("Frequency (Hz)"); self._spec_ax.set_ylabel("Power")
            self._spec_ax.set_title("FFT Power — Ch1 (Frontal)", fontsize=9)
            self._spec_bars = None
            spec_canvas = FigureCanvasTkAgg(self._spec_fig, master=spec_lf)
            spec_canvas.draw()
            spec_canvas.get_tk_widget().pack(fill=tk.X)
            self._spec_mpl = spec_canvas

        # ── prediction output ─────────────────────────────────────────────────
        pred_frame = frame(self, bg=BG2); pred_frame.pack(fill=tk.X, padx=14, pady=6)
        lbl(pred_frame, "Detected:", fg=FG2, bg=BG2, size=10).pack(side=tk.LEFT, padx=10, pady=8)
        self._pred_lbl = lbl(pred_frame, "—", fg=FG, bg=BG2, size=18, bold=True)
        self._pred_lbl.pack(side=tk.LEFT, padx=6)
        self._conf_lbl = lbl(pred_frame, "", fg=FG2, bg=BG2, size=9)
        self._conf_lbl.pack(side=tk.LEFT, padx=6)

        # confidence bars
        conf_frame = frame(self, bg=BG2); conf_frame.pack(fill=tk.X, padx=14, pady=(0, 8))
        self._conf_bars = {}
        for i, (name, color) in enumerate(zip(CLASS_NAMES, SSVEP_COLORS + [FG2])):
            fr = frame(conf_frame, bg=BG2); fr.pack(fill=tk.X, padx=10, pady=2)
            lbl(fr, name, fg=color, bg=BG2, size=8, width=8).pack(side=tk.LEFT)
            bg_bar = frame(fr, bg=BG3, height=12); bg_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            bar = frame(bg_bar, bg=color, height=12, width=2)
            bar.place(x=0, y=0, relheight=1)
            pct = tk.StringVar(value="0%")
            lbl(fr, "", fg=color, bg=BG2, size=8, textvariable=pct, width=5).pack(side=tk.LEFT, padx=4)
            self._conf_bars[i] = (bar, bg_bar, pct)

    def _build_rects(self):
        self._canvas.delete("all")
        self._rects.clear()
        self._freq_labels.clear()
        w = self._canvas.winfo_width() or 900
        h = self._canvas.winfo_height() or 240
        panel_w = w // 3
        for i in range(3):
            x0 = i * panel_w; x1 = x0 + panel_w
            r = self._canvas.create_rectangle(x0, 0, x1, h,
                                               fill=SSVEP_DARK[i], outline="")
            lbl_id = self._canvas.create_text(
                (x0 + x1) // 2, h - 20,
                text=f"{SSVEP_NAMES[i]}\n{SSVEP_FREQS[i]:.0f} Hz",
                fill=SSVEP_COLORS[i], font=("Segoe UI", 10, "bold"), justify="center")
            self._rects.append(r)
            self._freq_labels.append(lbl_id)

    def _on_resize(self, _):
        self._build_rects()

    def _toggle_stim(self):
        self._stim_on = not self._stim_on
        if self._stim_on:
            self._stim_btn.config(text="Stop Stimulus", bg=ERR)
        else:
            self._stim_btn.config(text="Start Stimulus", bg=OK)
            for i, r in enumerate(self._rects):
                self._canvas.itemconfig(r, fill=SSVEP_DARK[i])

    def _tick(self):
        if self._stim_on and self._rects:
            t = time.time()
            for i, (r, freq) in enumerate(zip(self._rects, SSVEP_FREQS)):
                phase = (t * freq) % 1.0
                color = SSVEP_COLORS[i] if phase < 0.5 else SSVEP_DARK[i]
                self._canvas.itemconfig(r, fill=color)
        self.after(14, self._tick)   # ~70 fps flicker loop

    def update_display(self):
        with LIVE.lock:
            sp = dict(LIVE.ssvep_p)
            sf = LIVE.spectrum_f.copy()
            spp = LIVE.spectrum_p.copy()

        # SSVEP spectrum plot
        if MPL_OK and len(sf) > 0 and len(spp) > 0:
            self._spec_ax.cla()
            self._spec_ax.set_facecolor(BG)
            self._spec_ax.plot(sf, spp, color=FG2, lw=0.8, alpha=0.6)
            # highlight SSVEP bins
            for freq, color in zip(SSVEP_FREQS, SSVEP_COLORS):
                mask = (sf >= freq - 0.6) & (sf <= freq + 0.6)
                if mask.any():
                    self._spec_ax.fill_between(sf, spp, where=mask, color=color, alpha=0.7)
                    peak = float(spp[mask].max())
                    self._spec_ax.text(freq, peak * 1.05, f"{freq:.0f}",
                                       ha="center", va="bottom", fontsize=7,
                                       color=color)
            self._spec_ax.set_xlabel("Hz"); self._spec_ax.set_ylabel("Power")
            self._spec_ax.set_title("FFT Power — Ch1", fontsize=8)
            for sp_obj in self._spec_ax.spines.values(): sp_obj.set_edgecolor(BG4)
            self._spec_ax.tick_params(colors=FG2)
            self._spec_mpl.draw_idle()

        # Prediction
        if PREDICTOR and PREDICTOR.ready:
            ep = LIVE.last_epoch
            if ep:
                cls, conf, probs = PREDICTOR.predict(ep)
                if cls >= 0:
                    name  = PREDICTOR.class_name(cls)
                    color = SSVEP_COLORS[cls] if cls < len(SSVEP_COLORS) else FG
                    self._pred_lbl.config(text=name, fg=color)
                    self._conf_lbl.config(text=f"confidence: {conf*100:.0f}%")
                else:
                    self._pred_lbl.config(text="—", fg=FG)
                    self._conf_lbl.config(text=f"low confidence ({conf*100:.0f}%)")

                # confidence bars
                for i, (bar, bg_bar, pct_var) in self._conf_bars.items():
                    p = float(probs[i]) if i < len(probs) else 0.0
                    bg_bar.update_idletasks()
                    w = bg_bar.winfo_width()
                    bar.place(x=0, y=0, relheight=1, width=max(2, int(w * p)))
                    pct_var.set(f"{p*100:.0f}%")
        else:
            self._pred_lbl.config(text="No model", fg=WARN)
            self._conf_lbl.config(text="Train a model first")


# ═══════════════════════════════════════════════════════════════════════════════
# Wing 3 — Prepare Model
# ═══════════════════════════════════════════════════════════════════════════════
class WingPrepare(tk.Frame):
    def __init__(self, nb):
        super().__init__(nb, bg=BG)
        self._calib_epochs  = {i: [] for i in range(N_CLASSES)}
        self._calib_active  = False
        self._calib_class   = 0
        self._calib_timer   = 0
        self._db_subjects   = []
        self._build()

    def _build(self):
        lbl(self, "Prepare Model — New Subject Calibration", fg=ACC, bg=BG,
            size=13, bold=True).pack(anchor="w", padx=14, pady=(10, 2))
        lbl(self, f"Collect {CALIB_EPOCHS_PER_CLASS} epochs per colour, then train.",
            fg=FG2, bg=BG, size=9).pack(anchor="w", padx=14)

        mongo_row = frame(self, bg=BG); mongo_row.pack(fill=tk.X, padx=14, pady=(2, 4))
        self._mongo_badge = lbl(mongo_row, "MongoDB: checking…", fg=FG3, bg=BG, size=8)
        self._mongo_badge.pack(side=tk.LEFT)

        # ── subject info ──────────────────────────────────────────────────────
        info_lf = tk.LabelFrame(self, text="  Subject Info  ", bg=BG2, fg=ACC,
                                 font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        info_lf.pack(fill=tk.X, padx=14, pady=8)

        r0 = frame(info_lf, bg=BG2); r0.pack(fill=tk.X, pady=3)
        lbl(r0, "Subject ID / Name:", fg=FG2, bg=BG2).pack(side=tk.LEFT)
        self._subj_var = tk.StringVar(value="subject_01")
        tk.Entry(r0, textvariable=self._subj_var, bg=BG3, fg=FG, width=24,
                 insertbackground=FG, relief=tk.FLAT,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)
        lbl(r0, "Notes:", fg=FG2, bg=BG2).pack(side=tk.LEFT, padx=(12, 4))
        self._notes_var = tk.StringVar()
        tk.Entry(r0, textvariable=self._notes_var, bg=BG3, fg=FG, width=40,
                 insertbackground=FG, relief=tk.FLAT,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)

        # ── calibration protocol ──────────────────────────────────────────────
        cal_lf = tk.LabelFrame(self, text="  Calibration Protocol  ", bg=BG2, fg=ACC,
                                font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        cal_lf.pack(fill=tk.X, padx=14, pady=4)

        instr_row = frame(cal_lf, bg=BG2); instr_row.pack(fill=tk.X, pady=4)
        self._instr_lbl = lbl(instr_row, "Press 'Start Calibration' to begin.",
                               fg=FG, bg=BG2, size=11, bold=True)
        self._instr_lbl.pack(side=tk.LEFT, padx=6)

        # big colour indicator
        self._color_box = tk.Canvas(cal_lf, bg=BG3, height=80, width=300,
                                     highlightthickness=1, highlightbackground=BG4)
        self._color_box.pack(pady=6)
        self._color_rect  = self._color_box.create_rectangle(0, 0, 300, 80,
                                                               fill=BG3, outline="")
        self._color_text  = self._color_box.create_text(
            150, 40, text="—", fill=FG, font=("Segoe UI", 16, "bold"))

        # countdown
        self._countdown_var = tk.StringVar(value="")
        lbl(cal_lf, "", fg=WARN, bg=BG2, size=20, bold=True,
            textvariable=self._countdown_var).pack()

        # progress per class
        prog_frame = frame(cal_lf, bg=BG2); prog_frame.pack(fill=tk.X, pady=6)
        self._prog_vars = {}
        self._prog_bars = {}
        for i, (name, color) in enumerate(zip(CLASS_NAMES, SSVEP_COLORS + [FG2])):
            fr = frame(prog_frame, bg=BG2); fr.pack(fill=tk.X, padx=6, pady=2)
            lbl(fr, name, fg=color, bg=BG2, size=8, width=8).pack(side=tk.LEFT)
            bg_bar = frame(fr, bg=BG3, height=10); bg_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            bar = frame(bg_bar, bg=color, height=10, width=2)
            bar.place(x=0, y=0, relheight=1)
            v = tk.StringVar(value="0"); lbl(fr, "", fg=color, bg=BG2, size=8,
                                              textvariable=v, width=8).pack(side=tk.LEFT, padx=4)
            self._prog_vars[i] = v
            self._prog_bars[i] = (bar, bg_bar)

        btn_row = frame(cal_lf, bg=BG2); btn_row.pack(fill=tk.X, pady=6)
        self._cal_btn = btn(btn_row, "Start Calibration", self._start_calib,
                             color=OK, fg_col=BG)
        self._cal_btn.pack(side=tk.LEFT, padx=4)
        btn(btn_row, "Reset", self._reset_calib, color=ERR, fg_col=FG).pack(
            side=tk.LEFT, padx=4)

        # ── training ──────────────────────────────────────────────────────────
        train_lf = tk.LabelFrame(self, text="  Train Model  ", bg=BG2, fg=ACC,
                                  font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        train_lf.pack(fill=tk.X, padx=14, pady=4)

        tr_row = frame(train_lf, bg=BG2); tr_row.pack(fill=tk.X, pady=4)
        self._train_btn = btn(tr_row, "Train Model", self._train_model,
                               color=ACC, fg_col=BG)
        self._train_btn.pack(side=tk.LEFT, padx=4)
        self._train_status = lbl(tr_row, "Collect calibration data first.",
                                  fg=FG2, bg=BG2, size=8)
        self._train_status.pack(side=tk.LEFT, padx=10)

        # ── load from MongoDB ─────────────────────────────────────────────────
        db_lf = tk.LabelFrame(self, text="  Load Saved Model from MongoDB  ",
                               bg=BG2, fg=ACC, font=("Segoe UI", 9, "bold"),
                               padx=8, pady=6)
        db_lf.pack(fill=tk.X, padx=14, pady=4)

        list_frame = frame(db_lf, bg=BG2); list_frame.pack(fill=tk.X, pady=4)
        self._db_list = tk.Listbox(list_frame, bg=BG3, fg=FG,
                                    selectbackground=ACC, selectforeground=BG,
                                    height=4, font=("Consolas", 8), activestyle="none")
        self._db_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._db_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._db_list.config(yscrollcommand=sb.set)

        db_btn_row = frame(db_lf, bg=BG2); db_btn_row.pack(fill=tk.X, pady=(2, 0))
        btn(db_btn_row, "Refresh List", self._refresh_db_list,
            color=BG4, fg_col=FG).pack(side=tk.LEFT, padx=4)
        btn(db_btn_row, "Load Selected Model", self._load_from_db,
            color=ACC, fg_col=BG).pack(side=tk.LEFT, padx=4)
        self._db_status = lbl(db_btn_row, "", fg=FG2, bg=BG2, size=8)
        self._db_status.pack(side=tk.LEFT, padx=8)

        # log
        self._log = scrolledtext.ScrolledText(self, height=5, bg=BG3, fg=FG2,
                                               font=("Consolas", 8),
                                               state="disabled", wrap=tk.WORD)
        self._log.pack(fill=tk.X, padx=14, pady=(4, 10))

        # kick off async checks after window renders
        self.after(600, self._check_mongo_status)
        self.after(1200, self._refresh_db_list)

    def _log_msg(self, msg, color=FG2):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end"); self._log.config(state="disabled")

    # ── calibration state machine ─────────────────────────────────────────────

    def _start_calib(self):
        if self._calib_active:
            self._calib_active = False
            self._cal_btn.config(text="Start Calibration", bg=OK)
            self._instr_lbl.config(text="Calibration paused.")
            return
        self._calib_active = True
        self._calib_class  = 0
        self._calib_timer  = 3   # 3-second countdown before first class
        self._cal_btn.config(text="Pause Calibration", bg=WARN)
        self._advance_class()

    def _advance_class(self):
        if not self._calib_active:
            return
        if self._calib_class >= N_CLASSES:
            self._calib_active = False
            self._cal_btn.config(text="Start Calibration", bg=OK)
            self._instr_lbl.config(text="Collection complete! Press Train Model.")
            self._color_box.itemconfig(self._color_rect, fill=BG3)
            self._color_box.itemconfig(self._color_text, text="Done", fill=OK)
            self._countdown_var.set("")
            return

        name  = CLASS_NAMES[self._calib_class]
        color = SSVEP_COLORS[self._calib_class] if self._calib_class < len(SSVEP_COLORS) else FG2
        self._instr_lbl.config(text=f"Focus on:  {name}  ({CALIB_EPOCHS_PER_CLASS} epochs needed)")
        self._color_box.itemconfig(self._color_rect, fill=color)
        self._color_box.itemconfig(self._color_text, text=name, fill="white")
        self._calib_timer = 3
        self._countdown()

    def _countdown(self):
        if not self._calib_active: return
        if self._calib_timer > 0:
            self._countdown_var.set(str(self._calib_timer))
            self._calib_timer -= 1
            self.after(1000, self._countdown)
        else:
            self._countdown_var.set("")
            self._collect_class()

    def _collect_class(self):
        if not self._calib_active: return
        ep = LIVE.last_epoch
        if ep:
            self._calib_epochs[self._calib_class].append(ep)
            n = len(self._calib_epochs[self._calib_class])
            self._prog_vars[self._calib_class].set(f"{n}/{CALIB_EPOCHS_PER_CLASS}")
            self._update_prog_bar(self._calib_class, n)
            if n >= CALIB_EPOCHS_PER_CLASS:
                self._log_msg(f"{CLASS_NAMES[self._calib_class]}: {n} epochs collected.")
                self._calib_class += 1
                self.after(500, self._advance_class)
                return
        self.after(1050, self._collect_class)   # wait ~1 epoch then retry

    def _update_prog_bar(self, idx, count):
        if idx not in self._prog_bars: return
        bar, bg_bar = self._prog_bars[idx]
        bg_bar.update_idletasks()
        w = bg_bar.winfo_width()
        frac = min(1.0, count / CALIB_EPOCHS_PER_CLASS)
        bar.place(x=0, y=0, relheight=1, width=max(2, int(w * frac)))

    def _reset_calib(self):
        self._calib_active = False
        self._calib_class  = 0
        self._calib_epochs = {i: [] for i in range(N_CLASSES)}
        for i in range(N_CLASSES):
            self._prog_vars[i].set("0")
            bar, bg_bar = self._prog_bars[i]
            bar.place(x=0, y=0, relheight=1, width=2)
        self._instr_lbl.config(text="Reset. Press 'Start Calibration' to begin.")
        self._color_box.itemconfig(self._color_rect, fill=BG3)
        self._color_box.itemconfig(self._color_text, text="—", fill=FG)
        self._countdown_var.set("")
        self._cal_btn.config(text="Start Calibration", bg=OK)

    def _train_model(self):
        all_eps, all_lbl = [], []
        for cls_idx, eps in self._calib_epochs.items():
            all_eps.extend(eps)
            all_lbl.extend([cls_idx] * len(eps))
        if len(all_eps) < MIN_EPOCHS_TO_TRAIN:
            messagebox.showinfo("Not enough data",
                                f"Need {MIN_EPOCHS_TO_TRAIN} epochs minimum. Collect more.")
            return
        subj  = self._subj_var.get().strip() or "unknown"
        notes = self._notes_var.get().strip()
        calib = {i: list(eps) for i, eps in self._calib_epochs.items()}
        self._train_status.config(text="Training…", fg=WARN)
        def _run():
            # ── build feature matrix for new epochs ───────────────────────────
            X_new = np.array([feature_vector(ep) for ep in all_eps], dtype=np.float32)
            y_new = np.array(all_lbl, dtype=np.int32)
            prior_info = ""

            # ── merge with prior sessions from MongoDB ────────────────────────
            if MONGO_STORE_OK:
                X_pr, y_pr, pr_msg = mongo_store.load_epoch_features(subj)
                if X_pr is not None and len(X_pr):
                    X_new = np.vstack([X_pr, X_new])
                    y_new = np.concatenate([y_pr, y_new])
                    prior_info = f"  (+{len(X_pr)} prior epochs — {pr_msg.split()[0]} total)"

            res = train_from_feature_arrays(X_new, y_new)

            # ── save new session epochs to MongoDB ────────────────────────────
            mongo_msg = ""
            mongo_ok  = False
            if res["ok"] and MONGO_STORE_OK:
                mongo_ok, mongo_msg = mongo_store.save_model(subj, notes, res, calib)

            def _done():
                color = OK if res["ok"] else ERR
                status_txt = res["msg"] + prior_info
                self._train_status.config(text=status_txt, fg=color)
                self._log_msg(f"[{subj}] {status_txt}")
                if MONGO_STORE_OK and res["ok"]:
                    self._log_msg(f"[MongoDB] {mongo_msg}", color=OK if mongo_ok else ERR)
                    if mongo_ok:
                        self.after(400, self._refresh_db_list)
                if res["ok"] and PREDICTOR:
                    PREDICTOR.set_model(res["model"], res["feature_dim"])
                    self._log_msg("Model loaded into predictor. Switch to Subject Testing.")
            self.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

    # ── MongoDB helpers ───────────────────────────────────────────────────────

    def _check_mongo_status(self):
        if not MONGO_STORE_OK:
            self._mongo_badge.config(text="MongoDB: pymongo not installed  (pip install pymongo)", fg=ERR)
            return
        def _run():
            ok, msg = mongo_store.ping()
            def _upd():
                dot = "●" if ok else "○"
                self._mongo_badge.config(text=f"MongoDB: {dot} {msg}", fg=OK if ok else ERR)
            self.after(0, _upd)
        threading.Thread(target=_run, daemon=True).start()

    def _refresh_db_list(self):
        if not MONGO_STORE_OK:
            return
        def _run():
            rows = mongo_store.list_subjects()
            def _upd():
                self._db_subjects = rows
                self._db_list.delete(0, tk.END)
                if not rows:
                    self._db_list.insert(tk.END, "  (no subjects found)")
                    return
                for r in rows:
                    line = (f"  {r['subject_id']:<20}  acc={r['accuracy']}  "
                            f"{r['total_epochs']} epochs  {r['last_trained']}")
                    self._db_list.insert(tk.END, line)
            self.after(0, _upd)
        threading.Thread(target=_run, daemon=True).start()

    def _load_from_db(self):
        if not MONGO_STORE_OK:
            messagebox.showinfo("MongoDB unavailable",
                                "pymongo not installed — run: pip install pymongo")
            return
        sel = self._db_list.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Click a subject row first."); return
        idx = sel[0]
        if not self._db_subjects or idx >= len(self._db_subjects):
            return
        subj = self._db_subjects[idx]["subject_id"]
        self._db_status.config(text=f"Loading '{subj}'…", fg=WARN)
        def _run():
            payload, msg = mongo_store.load_model_for_subject(subj)
            def _done():
                if payload is None:
                    self._db_status.config(text=msg, fg=ERR)
                    self._log_msg(f"[DB] {msg}")
                    return
                if PREDICTOR:
                    PREDICTOR.set_model(payload["model"], payload["feature_dim"])
                self._subj_var.set(subj)
                self._db_status.config(text=msg, fg=OK)
                self._log_msg(f"[DB] {msg}")
                self._log_msg("Model loaded into predictor. Switch to Subject Testing.")
            self.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

    def update_display(self):
        pass   # passive — driven by calibration state machine


# ═══════════════════════════════════════════════════════════════════════════════
# Header — connection status bar
# ═══════════════════════════════════════════════════════════════════════════════
class Header(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG2, height=44)
        self.pack_propagate(False)
        self._build()

    def _build(self):
        lbl(self, "  Neurosystem", fg=ACC, bg=BG2, size=13, bold=True).pack(
            side=tk.LEFT, padx=4, pady=8)
        lbl(self, "EEG Decision Studio  |  BioAmp EXG Pill + ESP32",
            fg=FG2, bg=BG2, size=8).pack(side=tk.LEFT, padx=4)

        right = frame(self, bg=BG2); right.pack(side=tk.RIGHT, padx=12)

        self._conn_dot = lbl(right, "●", fg=ERR, bg=BG2, size=11)
        self._conn_dot.pack(side=tk.LEFT)
        self._conn_txt = lbl(right, f" {ESP32_IP}", fg=FG2, bg=BG2, size=8)
        self._conn_txt.pack(side=tk.LEFT, padx=(2, 12))

        self._ep_lbl  = lbl(right, "Epoch —", fg=FG2, bg=BG2, size=8)
        self._ep_lbl.pack(side=tk.LEFT, padx=8)

        self._dom_lbl = lbl(right, "—", fg=FG2, bg=BG2, size=8)
        self._dom_lbl.pack(side=tk.LEFT, padx=8)

        self._model_lbl = lbl(right, "No model", fg=WARN, bg=BG2, size=8)
        self._model_lbl.pack(side=tk.LEFT, padx=8)

    def update(self, connected: bool, epoch_no: int, dom_band: str, dom_band2: str = "—"):
        self._conn_dot.config(fg=OK if connected else ERR)
        self._conn_txt.config(text=f" {ESP32_IP}  {'Connected' if connected else 'Reconnecting…'}",
                               fg=OK if connected else ERR)
        self._ep_lbl.config(text=f"Epoch {epoch_no}")
        self._dom_lbl.config(text=f"Ch1: {dom_band.upper()}  Ch2: {dom_band2.upper()}")
        if PREDICTOR and PREDICTOR.ready:
            self._model_lbl.config(text="Model: Ready", fg=OK)
        else:
            self._model_lbl.config(text="No model", fg=WARN)


# ═══════════════════════════════════════════════════════════════════════════════
# App — main window
# ═══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neurosystem — EEG Decision Studio")
        self.geometry("1340x840"); self.minsize(1000, 680)
        self.configure(bg=BG)
        self._style()
        self._build()
        self._start_ws()
        self._poll()   # start GUI update loop

    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                     padding=[18, 8], font=("Segoe UI", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", BG2)], foreground=[("selected", ACC)])
        s.configure("TScrollbar", background=BG3, troughcolor=BG,
                     arrowcolor=FG2, borderwidth=0)

    def _build(self):
        self._header = Header(self)
        self._header.pack(fill=tk.X, side=tk.TOP)

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X)

        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self._wing1 = WingAnalysis(nb)
        self._wing2 = WingSubject(nb)
        self._wing3 = WingPrepare(nb)

        nb.add(self._wing1, text="  Analysis & Testing  ")
        nb.add(self._wing2, text="  Subject Testing  ")
        nb.add(self._wing3, text="  Prepare Model  ")

    def _start_ws(self):
        if not WS_OK:
            self._ws = None
            return
        self._ws = WSReader()
        self._ws.on_connect_cb    = lambda: None
        self._ws.on_disconnect_cb = lambda: None
        self._ws.start()

    def _poll(self):
        """Main GUI loop — runs every 100 ms."""
        # Drain WebSocket queue into LIVE state
        if self._ws:
            for msg in self._ws.drain():
                LIVE.ingest(msg)

        # Determine dominant bands for both channels
        with LIVE.lock:
            rel  = dict(LIVE.band_rel)
            rel2 = dict(LIVE.band_rel2)
            ep   = LIVE.epoch_no
            connected = self._ws.connected if self._ws else False

        dom  = max(rel,  key=rel.get)  if any(rel.values())  else "—"
        dom2 = max(rel2, key=rel2.get) if any(rel2.values()) else "—"

        # Update all wings + header
        self._header.update(connected, ep, dom, dom2)
        self._wing1.update_display()
        self._wing2.update_display()

        self.after(100, self._poll)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not WS_OK:
        print("WARNING: websocket-client not installed.")
        print("  Run: pip install websocket-client")
    if not MODULES_OK:
        print(f"WARNING: Some modules failed to import: {_MODULE_ERR if not MODULES_OK else ''}")
        print("  Run: pip install numpy scipy scikit-learn matplotlib")

    app = App()
    app.mainloop()
