import time, csv, random, requests, sys
import tkinter as tk

ESP = "http://192.168.1.1"
COLORS = ["red","green","blue","yellow","magenta","cyan"]
TRIALS_PER = 20
CUE_SEC, IMAGINE_SEC, REST_SEC = 2.0, 4.0, 2.0
FS = 100
dt = 1.0/FS

root = tk.Tk(); root.attributes("-fullscreen", True)
canvas = tk.Canvas(root, bg="grey", highlightthickness=0); canvas.pack(fill="both", expand=True)
root.update()

rows = []
def get_eeg():
    try: d = requests.get(f"{ESP}/eeg", timeout=0.2).json()
    except: d = {"ch1":0,"ch2":0,"ch3":0,"ch4":0,"ch5":0,"ch6":0}
    return [d[f"ch{i}"] for i in range(1,7)]

seq = COLORS * TRIALS_PER
random.shuffle(seq)
start = time.time()

with open("eeg_imagine.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","phase","label","ch1","ch2","ch3","ch4","ch5","ch6"])
    for label in seq:
        # cue
        canvas.configure(bg=label); root.update()
        t0 = time.time()
        while time.time()-t0 < CUE_SEC:
            rows.append([time.time()-start,"cue",label,*get_eeg()])
            time.sleep(dt)
        # imagine (grey)
        canvas.configure(bg="grey"); root.update()
        t0 = time.time()
        while time.time()-t0 < IMAGINE_SEC:
            rows.append([time.time()-start,"imagine",label,*get_eeg()])
            time.sleep(dt)
        # rest
        canvas.configure(bg="black"); root.update()
        t0 = time.time()
        while time.time()-t0 < REST_SEC:
            rows.append([time.time()-start,"rest","none",*get_eeg()])
            time.sleep(dt)
    for r in rows: w.writerow(r)

canvas.configure(bg="black"); root.update(); time.sleep(0.5)
root.destroy()
print("Saved eeg_imagine.csv")
