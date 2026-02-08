from flask import Flask, jsonify
import numpy as np, requests, time, joblib, collections

ESP = "http://192.168.1.1"
FS = 100
WIN = 2.0
N = int(FS*WIN)

bufs = [collections.deque(maxlen=N) for _ in range(6)]
model = joblib.load("imagined_color_lda.joblib")

def poll():
    while True:
        try:
            d = requests.get(f"{ESP}/eeg", timeout=0.2).json()
            for i in range(6): bufs[i].append(d[f"ch{i+1}"])
        except: pass
        time.sleep(1.0/FS)

import threading; threading.Thread(target=poll, daemon=True).start()

def features():
    if any(len(b)<N for b in bufs): return None
    feats=[]
    for i in range(6):
        x = np.array(bufs[i], dtype=float)
        x = x - x.mean()
        win = np.hanning(len(x))
        sp = np.fft.rfft(x*win)
        freqs = np.fft.rfftfreq(len(x), 1/FS)
        psd = (np.abs(sp)**2)/(FS*np.sum(win**2))
        def bp(a,b):
            m=(freqs>=a)&(freqs<=b); 
            return np.log10(psd[m].sum()+1e-12)
        for a,b in [(1,4),(4,7),(8,13),(13,30),(30,45)]:
            feats.append(bp(a,b))
    return np.array(feats).reshape(1,-1)

app = Flask(__name__)

@app.route("/predict_imagined")
def predict_imagined():
    X = features()
    if X is None: return jsonify({"color":"unknown","confidence":0})
    proba = getattr(model, "predict_proba", lambda Z: None)(X)
    y = model.predict(X)[0]
    conf = float(proba.max()) if proba is not None else 0.0
    return jsonify({"color": y, "confidence": conf})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
