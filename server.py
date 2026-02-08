from flask import Flask, jsonify
import numpy as np, time, requests, threading, collections

ESP_IP = "http://192.168.1.1"   
FS = 100                        # assumed sampling rate via polling (Hz)
WINDOW_SEC = 3.0                # analysis window
N_SAMPLES = int(FS*WINDOW_SEC)

TARGETS = {  # color -> frequency Hz
    "red":8.0, "green":10.0, "blue":12.0,
    "yellow":15.0, "magenta":18.0, "cyan":20.0
}
BANDS = {c:(f-0.6, f+0.6) for c,f in TARGETS.items()}  # narrow bands

buf_ch5 = collections.deque(maxlen=N_SAMPLES)
buf_ch6 = collections.deque(maxlen=N_SAMPLES)

def poll_eeg():
    # crude 100 Hz polling (tune to your actual capability)
    dt = 1.0/FS
    while True:
        try:
            d = requests.get(f"{ESP_IP}/eeg", timeout=0.2).json()
            # Use occipital channels (adapt as needed)
            buf_ch5.append(d["ch5"])
            buf_ch6.append(d["ch6"])
        except Exception:
            pass
        time.sleep(dt)

threading.Thread(target=poll_eeg, daemon=True).start()

app = Flask(__name__)

def band_power(sig, fs, f1, f2):
    sig = np.array(sig, dtype=float)
    sig = sig - sig.mean()
    win = np.hanning(len(sig))
    sp = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    psd = (np.abs(sp)**2) / (fs * np.sum(win**2))
    band = (freqs >= f1) & (freqs <= f2)
    return psd[band].sum(), freqs[band][np.argmax(psd[band])] if band.any() else (0, 0)

@app.route("/predict_ssvep")
def predict_ssvep():
    if len(buf_ch5) < N_SAMPLES or len(buf_ch6) < N_SAMPLES:
        return jsonify({"color": "unknown", "peak_hz": 0.0, "confidence": 0.0})

    x = (np.array(buf_ch5) + np.array(buf_ch6))/2.0

    powers, peaks = {}, {}
    for color,(f1,f2) in BANDS.items():
        p, pk = band_power(x, FS, f1, f2)
        powers[color] = p
        peaks[color] = pk

    best = max(powers, key=powers.get)
    # simple confidence: best / sum
    total = sum(powers.values()) + 1e-9
    conf = float(powers[best]/total)

    return jsonify({"color": best, "peak_hz": float(peaks[best]), "confidence": conf})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
