import pandas as pd, numpy as np, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score

FS = 100
WINDOW = 2.0  # use 2 s imagination windows
SAMPLES = int(WINDOW*FS)

def bandpower(x, fs, f1, f2):
    x = x - x.mean()
    win = np.hanning(len(x))
    sp = np.fft.rfft(x*win)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    psd = (np.abs(sp)**2)/(fs*np.sum(win**2))
    band = (freqs>=f1)&(freqs<=f2)
    return psd[band].sum()

df = pd.read_csv("eeg_imagine.csv")
df = df[df.phase=="imagine"].copy()

# Segment into non-overlapping 2 s windows per trial by label
features, labels = [], []
for lab, g in df.groupby((df.phase.ne(df.phase.shift()) | df.label.ne(df.label.shift())).cumsum()):
    seg = g.copy()
    if seg.phase.iloc[0]!="imagine": continue
    # chop into 2 s
    arr = seg[["ch1","ch2","ch3","ch4","ch5","ch6"]].to_numpy()
    nwin = arr.shape[0]//SAMPLES
    for k in range(nwin):
        X = arr[k*SAMPLES:(k+1)*SAMPLES, :]
        feats = []
        for ch in range(6):
            x = X[:,ch].astype(float)
            for (a,b) in [(1,4),(4,7),(8,13),(13,30),(30,45)]:
                feats.append(np.log10(bandpower(x, FS, a, b)+1e-12))
        features.append(feats)
        labels.append(seg.label.iloc[0])

X = np.array(features); y = np.array(labels)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", LinearDiscriminantAnalysis())])
pipe.fit(X_tr, y_tr)
print("Holdout acc:", pipe.score(X_te, y_te))
cv = cross_val_score(pipe, X, y, cv=5)
print("CV acc:", cv.mean(), "+/-", cv.std())

joblib.dump(pipe, "imagined_color_lda.joblib")
print("Saved imagined_color_lda.joblib")
