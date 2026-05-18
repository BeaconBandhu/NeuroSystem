"""
EEG ML Visualizer  —  BioAmp EXG Pill + ESP32  (WebSocket edition)
====================================================================
Connects to ws://192.168.1.33:81
Dual channel:  CH1 — Frontal  Fp1-Fp2 (GPIO 34)
               CH2 — Temporal T7-T8   (GPIO 35)

Keys (click the window first):
  R     — start / stop recording to CSV
  1-5   — set label: 1=Relaxed 2=Focused 3=Neutral 4=EyesOpen 5=EyesClosed
  C     — clear trend history
  Q     — quit
"""

import sys, threading, csv, datetime, os, json
import websocket
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
WS_URL      = "ws://192.168.1.33:81"
TREND_LEN   = 120           # rolling epochs (120 s)
EMA_ALPHA   = 0.3           # bar-chart smoothing
ARTIFACT_UV = 150.0         # filt_rms threshold for artifact flag

BAND_KEYS  = ["d", "t", "a", "b", "g"]
BAND_NAMES = ["Delta\n0.5-4Hz", "Theta\n4-8Hz", "Alpha\n8-13Hz",
              "Beta\n13-30Hz",  "Gamma\n30-45Hz"]
BAND_COLS  = ["#8C50FF", "#50B4FF", "#FFC832", "#3CDC64", "#FF5050"]
LABELS     = ["Relaxed", "Focused", "Neutral", "EyesOpen", "EyesClosed"]
QUAL_TEXT  = {"good": "GOOD", "floating": "FLOATING", "flat": "FLAT"}
QUAL_COL   = {"good": "#3CDC64", "floating": "#FFC832", "flat": "#B478FF"}

# ── Shared state ──────────────────────────────────────────────────────────────
lock  = threading.Lock()
state = dict(
    ep=0, ms=0,
    # Ch1
    Q="flat", dom=0, artifact=False,
    smooth=[0.0]*5,
    AB=0.0, TA=0.0, Frms=0.0, Rrms=0.0, Hum=0.0,
    # Ch2
    Q2="flat", artifact2=False,
    smooth2=[0.0]*5,
    AB2=0.0, TA2=0.0, Frms2=0.0,
    got_data=False,
    connected=False,
)
trend    = [deque([0.0]*TREND_LEN, maxlen=TREND_LEN) for _ in range(5)]
trend2   = [deque([0.0]*TREND_LEN, maxlen=TREND_LEN) for _ in range(5)]
trend_AB  = deque([0.0]*TREND_LEN, maxlen=TREND_LEN)
trend_TA  = deque([0.0]*TREND_LEN, maxlen=TREND_LEN)
trend_AB2 = deque([0.0]*TREND_LEN, maxlen=TREND_LEN)
trend_TA2 = deque([0.0]*TREND_LEN, maxlen=TREND_LEN)

rec   = dict(active=False, writer=None, file=None,
             saved=0, skipped=0, fname="")
label = [2]

_sm1 = [0.0]*5
_sm2 = [0.0]*5

# ── WebSocket message handler ─────────────────────────────────────────────────
def process_msg(msg: dict):
    rel1 = [msg.get("rel",  {}).get(k, 0.0) for k in BAND_KEYS]
    rel2 = [msg.get("rel2", {}).get(k, 0.0) for k in BAND_KEYS]
    bp1  = msg.get("bp",  {})
    bp2  = msg.get("bp2", {})

    for i in range(5):
        _sm1[i] = EMA_ALPHA * rel1[i] + (1 - EMA_ALPHA) * _sm1[i]
        _sm2[i] = EMA_ALPHA * rel2[i] + (1 - EMA_ALPHA) * _sm2[i]

    dom1  = rel1.index(max(rel1)) if any(rel1) else 0
    q1    = msg.get("quality",  "flat")
    q2    = msg.get("quality2", "flat")
    fr1   = float(msg.get("filt_rms",  0))
    fr2   = float(msg.get("filt_rms2", 0))
    art1  = fr1 > ARTIFACT_UV
    art2  = fr2 > ARTIFACT_UV

    with lock:
        state.update(
            ep=msg.get("epoch", 0),
            ms=msg.get("ts",    0),
            Q=q1,  dom=dom1, artifact=art1,
            Q2=q2,            artifact2=art2,
            smooth=_sm1[:],  smooth2=_sm2[:],
            AB=float(msg.get("ab_ratio",  0)),
            TA=float(msg.get("ta_ratio",  0)),
            AB2=float(msg.get("ab_ratio2", 0)),
            TA2=float(msg.get("ta_ratio2", 0)),
            Frms=fr1, Rrms=float(msg.get("raw_rms", 0)),
            Hum=float(msg.get("hum_pct", 0)), Frms2=fr2,
            got_data=True, connected=True,
        )
        for i in range(5):
            trend[i].append(rel1[i])
            trend2[i].append(rel2[i])
        trend_AB.append(float(msg.get("ab_ratio",  0)))
        trend_TA.append(float(msg.get("ta_ratio",  0)))
        trend_AB2.append(float(msg.get("ab_ratio2", 0)))
        trend_TA2.append(float(msg.get("ta_ratio2", 0)))

    # recording — good + non-artifact Ch1 epochs only
    if rec["active"]:
        if q1 == "good" and not art1:
            rec["writer"].writerow([
                msg.get("epoch",""), msg.get("ts",""),
                LABELS[label[0]], q1, q2,
                msg.get("raw_rms",""), fr1, fr2, msg.get("hum_pct",""),
                bp1.get("d",""), bp1.get("t",""), bp1.get("a",""),
                bp1.get("b",""), bp1.get("g",""),
                *[f"{r:.4f}" for r in rel1],
                msg.get("ab_ratio",""),  msg.get("ta_ratio",""),
                bp2.get("d",""), bp2.get("t",""), bp2.get("a",""),
                bp2.get("b",""), bp2.get("g",""),
                *[f"{r:.4f}" for r in rel2],
                msg.get("ab_ratio2",""), msg.get("ta_ratio2",""),
            ])
            rec["file"].flush()
            rec["saved"] += 1
        else:
            rec["skipped"] += 1


# ── WebSocket reader thread ───────────────────────────────────────────────────
def ws_reader():
    def on_open(app):
        print(f"Connected to {WS_URL}")
        with lock: state["connected"] = True

    def on_message(app, raw):
        try:
            process_msg(json.loads(raw))
        except Exception as e:
            print("Parse error:", e)

    def on_error(app, err):
        print(f"WS error: {err}")

    def on_close(app, code, msg):
        print("WS disconnected — reconnecting…")
        with lock: state["connected"] = False

    print(f"Connecting to {WS_URL}…")
    app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.run_forever(reconnect=3)


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9), facecolor="#0F0F19")
fig.canvas.manager.set_window_title(
    "EEG Visualizer  |  CH1: Fp1-Fp2  ·  CH2: T7-T8  |  ws://192.168.1.33:81")

gs = fig.add_gridspec(4, 2,
    left=0.05, right=0.98, top=0.91, bottom=0.06,
    hspace=0.60, wspace=0.28,
    height_ratios=[1.6, 1.0, 0.65, 0.40])

ax_bar1 = fig.add_subplot(gs[0, 0])   # CH1 band power bars
ax_bar2 = fig.add_subplot(gs[0, 1])   # CH2 band power bars
ax_tr1  = fig.add_subplot(gs[1, 0])   # CH1 trend
ax_tr2  = fig.add_subplot(gs[1, 1])   # CH2 trend
ax_rat1 = fig.add_subplot(gs[2, 0])   # CH1 ratios
ax_rat2 = fig.add_subplot(gs[2, 1])   # CH2 ratios
ax_ctrl = fig.add_subplot(gs[3, :])   # control bar (full width)

BG, PANEL = "#0F0F19", "#16162A"
for ax in [ax_bar1, ax_bar2, ax_tr1, ax_tr2, ax_rat1, ax_rat2, ax_ctrl]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color("#333355")

# ── CH1 band-power bars ───────────────────────────────────────────────────────
bars1 = ax_bar1.bar(BAND_NAMES, [0]*5, color=BAND_COLS,
                    width=0.6, edgecolor="#222244", linewidth=0.8)
ax_bar1.set_ylim(0, 1)
ax_bar1.set_ylabel("Relative power", color="#AAAACC", fontsize=9)
ax_bar1.set_title("CH1 — Frontal (Fp1-Fp2)  Band Power",
                  color="#CCCCEE", fontsize=10, pad=4)
ax_bar1.tick_params(colors="#AAAACC", labelsize=8)
ax_bar1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
pct1 = [ax_bar1.text(i, 0.01, "", ha="center", va="bottom",
                     color="white", fontsize=9, fontweight="bold")
        for i in range(5)]

# ── CH2 band-power bars ───────────────────────────────────────────────────────
bars2 = ax_bar2.bar(BAND_NAMES, [0]*5, color=BAND_COLS,
                    width=0.6, edgecolor="#222244", linewidth=0.8)
ax_bar2.set_ylim(0, 1)
ax_bar2.set_ylabel("Relative power", color="#AAAACC", fontsize=9)
ax_bar2.set_title("CH2 — Temporal (T7-T8)  Band Power",
                  color="#CCCCEE", fontsize=10, pad=4)
ax_bar2.tick_params(colors="#AAAACC", labelsize=8)
ax_bar2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
pct2 = [ax_bar2.text(i, 0.01, "", ha="center", va="bottom",
                     color="white", fontsize=9, fontweight="bold")
        for i in range(5)]

# ── CH1 trend lines ───────────────────────────────────────────────────────────
tr1_ln = []
for nm, col in zip(BAND_NAMES, BAND_COLS):
    ln, = ax_tr1.plot([], [], color=col, lw=1.5, label=nm.split("\n")[0], alpha=0.9)
    tr1_ln.append(ln)
ax_tr1.set_xlim(0, TREND_LEN); ax_tr1.set_ylim(0, 1)
ax_tr1.set_title("CH1 Relative Band Trend  (last 120 s)", color="#CCCCEE", fontsize=10, pad=4)
ax_tr1.set_ylabel("Relative power", color="#AAAACC", fontsize=9)
ax_tr1.tick_params(colors="#AAAACC", labelsize=8)
ax_tr1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax_tr1.legend(loc="upper left", fontsize=7, facecolor=BG,
              labelcolor="white", framealpha=0.7, ncol=5)

# ── CH2 trend lines ───────────────────────────────────────────────────────────
tr2_ln = []
for nm, col in zip(BAND_NAMES, BAND_COLS):
    ln, = ax_tr2.plot([], [], color=col, lw=1.5, label=nm.split("\n")[0], alpha=0.9)
    tr2_ln.append(ln)
ax_tr2.set_xlim(0, TREND_LEN); ax_tr2.set_ylim(0, 1)
ax_tr2.set_title("CH2 Relative Band Trend  (last 120 s)", color="#CCCCEE", fontsize=10, pad=4)
ax_tr2.set_ylabel("Relative power", color="#AAAACC", fontsize=9)
ax_tr2.tick_params(colors="#AAAACC", labelsize=8)
ax_tr2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax_tr2.legend(loc="upper left", fontsize=7, facecolor=BG,
              labelcolor="white", framealpha=0.7, ncol=5)

# ── CH1 ratio trends ──────────────────────────────────────────────────────────
ab1_ln, = ax_rat1.plot([], [], color="#FFC832", lw=1.5, label="Alpha/Beta")
ta1_ln, = ax_rat1.plot([], [], color="#50B4FF", lw=1.5, label="Theta/Alpha")
ax_rat1.set_xlim(0, TREND_LEN); ax_rat1.set_ylim(0, 5)
ax_rat1.set_title("CH1 Ratios  (A/B · T/A)", color="#CCCCEE", fontsize=10, pad=4)
ax_rat1.tick_params(colors="#AAAACC", labelsize=8)
ax_rat1.legend(loc="upper left", fontsize=8, facecolor=BG,
               labelcolor="white", framealpha=0.7)

# ── CH2 ratio trends ──────────────────────────────────────────────────────────
ab2_ln, = ax_rat2.plot([], [], color="#FFC832", lw=1.5, label="Alpha/Beta")
ta2_ln, = ax_rat2.plot([], [], color="#50B4FF", lw=1.5, label="Theta/Alpha")
ax_rat2.set_xlim(0, TREND_LEN); ax_rat2.set_ylim(0, 5)
ax_rat2.set_title("CH2 Ratios  (A/B · T/A)", color="#CCCCEE", fontsize=10, pad=4)
ax_rat2.tick_params(colors="#AAAACC", labelsize=8)
ax_rat2.legend(loc="upper left", fontsize=8, facecolor=BG,
               labelcolor="white", framealpha=0.7)

# ── Control bar ───────────────────────────────────────────────────────────────
ax_ctrl.axis("off")
ctrl_txt = ax_ctrl.text(0.5, 0.5, "",
    transform=ax_ctrl.transAxes,
    ha="center", va="center",
    color="#AAAACC", fontsize=9, fontfamily="monospace")

# ── Header + quality badges ───────────────────────────────────────────────────
fig.text(0.5, 0.965,
    "BioAmp EEG  |  CH1: Fp1-Fp2 Frontal  ·  CH2: T7-T8 Temporal"
    "  |  Fs=256 Hz, Epoch=1 s  |  ws://192.168.1.33:81",
    ha="center", color="#CCCCEE", fontsize=11)

q1_badge = fig.text(0.01, 0.965, "  CH1: ---  ",
    color="#888888", fontsize=10, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec="#333355"))
q2_badge = fig.text(0.22, 0.965, "  CH2: ---  ",
    color="#888888", fontsize=10, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec="#333355"))

xs = list(range(TREND_LEN))


# ── Animation update ──────────────────────────────────────────────────────────
def update(_):
    with lock:
        s    = dict(state)
        tr1  = [list(trend[i])  for i in range(5)]
        tr2  = [list(trend2[i]) for i in range(5)]
        ab1  = list(trend_AB);  ta1 = list(trend_TA)
        ab2  = list(trend_AB2); ta2 = list(trend_TA2)

    if not s["got_data"]:
        conn = "Connected — waiting for first epoch…" if s["connected"] else f"Connecting to {WS_URL}…"
        ctrl_txt.set_text(f"{conn}   Keys: R=record  1-5=label  C=clear  Q=quit")
        return

    sm1  = s["smooth"]
    sm2  = s["smooth2"]
    dom1 = s["dom"]
    dom2 = sm2.index(max(sm2)) if any(sm2) else 0

    # CH1 bars
    for i, bar in enumerate(bars1):
        bar.set_height(sm1[i])
        bar.set_alpha(1.0 if i == dom1 else 0.5)
        bar.set_edgecolor("white" if i == dom1 else "#222244")
        bar.set_linewidth(1.8 if i == dom1 else 0.8)
        pct1[i].set_position((i, sm1[i] + 0.01))
        pct1[i].set_text(f"{sm1[i]*100:.1f}%")
        pct1[i].set_color(BAND_COLS[i])

    # CH2 bars
    for i, bar in enumerate(bars2):
        bar.set_height(sm2[i])
        bar.set_alpha(1.0 if i == dom2 else 0.5)
        bar.set_edgecolor("white" if i == dom2 else "#222244")
        bar.set_linewidth(1.8 if i == dom2 else 0.8)
        pct2[i].set_position((i, sm2[i] + 0.01))
        pct2[i].set_text(f"{sm2[i]*100:.1f}%")
        pct2[i].set_color(BAND_COLS[i])

    # CH1 trends
    for i, ln in enumerate(tr1_ln):
        ln.set_data(xs, tr1[i])
        ln.set_linewidth(2.2 if i == dom1 else 0.9)
        ln.set_alpha(1.0  if i == dom1 else 0.45)

    # CH2 trends
    for i, ln in enumerate(tr2_ln):
        ln.set_data(xs, tr2[i])
        ln.set_linewidth(2.2 if i == dom2 else 0.9)
        ln.set_alpha(1.0  if i == dom2 else 0.45)

    # CH1 ratios
    ab1_ln.set_data(xs, ab1); ta1_ln.set_data(xs, ta1)
    pk1 = max(max(ab1 + ta1, default=0) * 1.2, 1.0)
    ax_rat1.set_ylim(0, pk1)

    # CH2 ratios
    ab2_ln.set_data(xs, ab2); ta2_ln.set_data(xs, ta2)
    pk2 = max(max(ab2 + ta2, default=0) * 1.2, 1.0)
    ax_rat2.set_ylim(0, pk2)

    # Quality badges
    def fmt_badge(q, art, name):
        label_txt = QUAL_TEXT.get(q, "---")
        color     = QUAL_COL.get(q, "#888888")
        suffix    = " ARTIFACT" if art else ""
        return f"  {name}: {label_txt}{suffix}  ", color

    t1, c1 = fmt_badge(s["Q"],  s["artifact"],  "CH1")
    t2, c2 = fmt_badge(s["Q2"], s["artifact2"], "CH2")
    q1_badge.set_text(t1); q1_badge.set_color(c1)
    q1_badge.get_bbox_patch().set_edgecolor(c1)
    q2_badge.set_text(t2); q2_badge.set_color(c2)
    q2_badge.get_bbox_patch().set_edgecolor(c2)

    # Control bar
    dom1_name = BAND_NAMES[dom1].split("\n")[0]
    dom2_name = BAND_NAMES[dom2].split("\n")[0]
    ep_info = (
        f"Ep#{s['ep']}  {s['ms']/1000:.1f}s  |  "
        f"CH1 dom:{dom1_name}  AB:{s['AB']:.2f}  TA:{s['TA']:.2f}"
        f"  Frms:{s['Frms']:.1f}µV  Hum:{s['Hum']:.0f}%  |  "
        f"CH2 dom:{dom2_name}  AB:{s['AB2']:.2f}  TA:{s['TA2']:.2f}"
        f"  Frms:{s['Frms2']:.1f}µV"
    )
    if rec["active"]:
        ctrl_txt.set_text(
            f"[REC]  {rec['fname']}  {rec['saved']} saved  {rec['skipped']} skip  "
            f"Label:{LABELS[label[0]]}  |  {ep_info}  |  R=stop  1-5=label  Q=quit")
        ctrl_txt.set_color("#FF6060")
    else:
        tail = f"  Last:{rec['saved']}ep" if rec["saved"] else ""
        ctrl_txt.set_text(
            f"R=record{tail}  Label:{LABELS[label[0]]}  |  {ep_info}  |  1-5=label  C=clear  Q=quit")
        ctrl_txt.set_color("#AAAACC")


# ── Key bindings ──────────────────────────────────────────────────────────────
def toggle_rec():
    if not rec["active"]:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.dirname(os.path.abspath(__file__))
        fn     = os.path.join(folder, f"eeg_data_{ts}.csv")
        f      = open(fn, "w", newline="")
        w      = csv.writer(f)
        w.writerow([
            "epoch", "ms", "label", "quality", "quality2",
            "raw_rms", "filt_rms", "filt_rms2", "hum_pct",
            "D_abs",  "T_abs",  "A_abs",  "B_abs",  "G_abs",
            "D_rel",  "T_rel",  "A_rel",  "B_rel",  "G_rel",
            "A_B_ratio", "T_A_ratio",
            "D2_abs", "T2_abs", "A2_abs", "B2_abs", "G2_abs",
            "D2_rel", "T2_rel", "A2_rel", "B2_rel", "G2_rel",
            "A2_B2_ratio", "T2_A2_ratio",
        ])
        rec.update(active=True, writer=w, file=f,
                   saved=0, skipped=0, fname=os.path.basename(fn))
        print(f"Recording started -> {fn}")
    else:
        rec["file"].close()
        print(f"Recording stopped. {rec['saved']} epochs -> {rec['fname']}")
        rec["active"] = False


def on_key(event):
    k = event.key
    if k in ("r", "R"):
        toggle_rec()
    elif k in "12345":
        label[0] = int(k) - 1
        print("Label:", LABELS[label[0]])
    elif k in ("c", "C"):
        for d in trend:  d.clear(); d.extend([0.0]*TREND_LEN)
        for d in trend2: d.clear(); d.extend([0.0]*TREND_LEN)
        for d in (trend_AB, trend_TA, trend_AB2, trend_TA2):
            d.clear(); d.extend([0.0]*TREND_LEN)
    elif k in ("q", "Q"):
        if rec["active"]: toggle_rec()
        plt.close("all"); sys.exit(0)


fig.canvas.mpl_connect("key_press_event", on_key)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    th = threading.Thread(target=ws_reader, daemon=True)
    th.start()
    ani = FuncAnimation(fig, update, interval=500, cache_frame_data=False)
    plt.show()
