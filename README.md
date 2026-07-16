# NeuroSystem

### From Neural Signals to Real-Time Target Acquisition

Built by **Aranya Bandhu** | Araxys Aerospace

---

## What is NeuroSystem?

NeuroSystem is an EEG-based AI system that translates raw human neural signals into real-time target acquisition commands — faster than conscious human reaction time.

The core insight comes from Libet's Experiment: the brain generates a readiness potential — a measurable electrical signal — 300 to 500 milliseconds before a person is consciously aware of their own decision to act. NeuroSystem detects and acts on this pre-conscious signal, compressing the human-to-machine decision loop to under 50ms.

The pilot thinks. The system fires. Before the pilot knows they have decided.

NeuroSystem is embedded within the Samridh fighter pilot helmet system, funded by the Indian Ministry of Defence under a 15 Lakh MSME grant.

---

## The Problem

Modern fighter pilots face a brutal constraint — aerial combat decisions that take even half a second can cost lives. The bottleneck is not the aircraft. It is not the weapon system. It is human reaction time.

Conventional interfaces require a conscious decision, then a physical input, then a system response. NeuroSystem eliminates the middle two steps entirely.

---

## Real Hardware. Real Signals. Built from Scratch.

This was not built in a lab with expensive equipment. It was built in a room with a consumer-grade BioAmp EEG headset, real electrodes, and real neural data collected from live subjects.

### EEG Hardware Setup

![ EEG Hardware](WhatsApp%20Image%202026-06-07%20at%206.48.29%20AM.jpeg)

![ EEG Hardware Close](WhatsApp%20Image%202026-06-07%20at%206.48.29%20AM%20(1).jpeg)

The headset uses dry electrodes placed at Fp1, Fp2, and reference positions to capture frontal lobe activity — the region most active during motor intent and target selection. Colored wires carry raw analog signals to the BioAmp EXG Pill + ESP32 microcontroller for digitization at 115200 baud.

---

## Signal Pipeline — Raw to Clean

### NeuroSystem Training Studio

![EEG Signal Cleaning Pipeline](Screenshot%202026-04-16%20020059.png)

The top waveform shows raw EEG data before cleaning — 23.1% of samples flagged as artifact. Red dots mark amplitude violations above 150μV. The signal is brutal in real-world environments: movement artifacts, EMG bleed, and electrical interference from nearby electronics.

The bottom waveform in green shows the signal after the three-stage cleaning pipeline:

- Amplitude threshold rejection — removes samples above 150μV
- IQR outlier multiplier — rejects statistical outliers beyond Q1/Q3 ± 3x IQR
- Delta and Beta artifact threshold — rejects entire 500ms batches where power exceeds 3x median

Result: clean, usable signal ready for CNN inference.

---

## Live Frequency Band Monitoring

![Live EEG Band Power Serial Monitor](WhatsApp%20Image%202026-04-16%20at%202.21.45%20AM.jpeg)

This is real-time band power streaming from the ESP32 over COM3 at 115200 baud. Each colored line represents a frequency band being monitored simultaneously:

- D (Delta) — slow wave baseline, dominant at rest
- T (Theta) — cognitive engagement
- A (Alpha) — suppressed during active motor intent
- B (Beta) — spikes during motor preparation, the primary trigger signal
- G (Gamma) — high-frequency target selection activity

The sharp spikes visible in the Delta and Beta bands correspond to motor intent events — these are the signals NeuroSystem's CNN is trained to detect and act on.

---

## Architecture


┌─────────────────────────────────────────────────────────────────────┐
│                        EEG ELECTRODE ARRAY                          │
│              (Real-world signal ingestion — noisy env)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SIGNAL PREPROCESSING                            │
│  • Notch Filter (50Hz) — removes electrical mains interference      │
│  • Bandpass Filter (0.1–40Hz) — isolates motor intent bands         │
│  • Artifact Rejection — discards epochs with movement noise         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FREQUENCY BAND EXTRACTION                        │
│  • Alpha  (8–13 Hz)  — baseline idle, suppressed during planning    │
│  • Beta   (13–30 Hz) — motor preparation & ERD detection            │
│  • Gamma  (30–100Hz) — high-level cognitive target selection        │
│  • SCP    (<1 Hz)    — Libet readiness potential detection          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CNN INFERENCE ENGINE                             │
│  • Lightweight CNN — <5ms latency, readiness potential detection    │
│  • Heavy validation CNN — full accuracy for target confirmation     │
│  • Ensemble cross-validation — no single model triggers action      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              GATE-LOCK VERIFICATION LAYER (Kernel Level)            │
│  • State hash computed every 5 inference cycles                     │
│  • Protected memory segment stores verified checkpoint              │
│  • Tamper detection blocks corrupted/drifted output                 │
│  • Auto-restores last verified state if integrity fails             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  REAL-TIME TARGET ACQUISITION                       │
│              End-to-end latency: < 50ms                             │
│         Decision made before conscious awareness                    │
└─────────────────────────────────────────────────────────────────────┘




---

## The Libet Experiment

Benjamin Libet discovered that the brain generates a readiness potential 300 to 500 milliseconds before a person consciously decides to act. Conscious intention is a lagging indicator of what the brain has already decided.

NeuroSystem trains on the pre-conscious signal, not the conscious one.

| Stage | Traditional Interface | NeuroSystem |
|---|---|---|
| Neural intent | Not detected | Readiness potential at -400ms |
| System begins | Waits for conscious input | Inference starts at -400ms |
| Action triggered | 600ms+ after intent | Under 50ms from RP onset |

---

## Technical Stack

| Category | Technology |
|---|---|
| Deep Learning | PyTorch, TensorFlow, Keras |
| Signal Processing | Pandas, Scikit-Learn, SciPy |
| Hardware | BioAmp EXG Pill + ESP32 |
| Embedded Deployment | TensorFlow Lite INT8 quantized |
| Backend | Flask, REST APIs |
| AI Tooling | Claude Code |
| OS | Linux, Embedded AI hardware |

---

## Gate-Lock Architecture

NeuroSystem introduced the Gate-Lock — a kernel-level verification system for high-reliability AI inference pipelines.

Every 5 inference cycles, a hash of the system state is locked to a protected memory segment. If any downstream process produces output that does not match the verified state — whether from drift, tampering, or corruption — Gate-Lock blocks it and restores the last clean checkpoint automatically.

This makes the entire inference pipeline tamper-resistant and self-correcting at the hardware level, not just in software.

Gate-Lock was later applied to LLM context management in BashIn, reducing token overhead by over 60% in extended agentic sessions.

---

## Results

| Metric | Result |
|---|---|
| End-to-end latency | Under 50ms |
| Decision advantage | 300–500ms over conscious reaction time |
| Competition | 1st Prize — AI Cohort 2025 |
| Funding | 50,000 INR |
| Research paper | Published |
| Patent | Filed |
| Selection | APF 2025 |

---

## Research & Publications

**NeuroSystem: From Neural Signals to Real-Time Target Acquisition**
Explored EEG-based neural intent translation into real-time AI-controlled actions. Supported rapid R&D execution, patent drafting, and full system validation.

**H3-D3-Py: A 3D Hexagonal Mesh Algorithm for Accurate Path Finding and Target Tracking**
Companion algorithm used in the broader Samridh defense system. Applicable to autonomous systems, aerospace navigation, and real-time tracking.

---

## Part of the Samridh Ecosystem

NeuroSystem is the neural interface layer of the Samridh AI-powered fighter pilot helmet, funded by the Indian Ministry of Defence.

Samridh System includes: 360 degree Situational Awareness Engine, Real-Time Decision Support AI, NeuroSystem (this repo), H3-D3-Py Path Planning Algorithm, and Embedded AI Deployment Layer.

3 research papers published. 3 patents filed. 15 Lakh INR government grant awarded.

---

## Notice

Certain implementation details, datasets, and deployment configurations are withheld due to the classified nature of the Samridh program under the Indian Ministry of Defence. This repository contains the research architecture, core algorithms, published findings, and real signal data only.

---

## Author

***Aranya Bandhu***
AI & ML Engineer | Araxys Aerospace | CMR Institute of Technology, Bengaluru

aranyabandhu2004@gmail.com | linkedin.com/in/aranyabandhu | bashin.live | github.com/BeaconBandhu

---

*"The brain decides before you know you've decided. NeuroSystem acts before you know you've decided."*
