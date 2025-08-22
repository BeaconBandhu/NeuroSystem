# NeuroSystem

This project explores brain–computer interfacing (BCI) using a 6-channel EEG setup with an ESP32. The system streams EEG signals over Wi-Fi, visualizes them in real time, and applies machine learning to predict the color a user is thinking of.

🔑 Key Features

ESP32-based EEG Acquisition

Reads 6 EEG channels (ADC pins 32–39).

Serves raw EEG values as JSON via /eeg endpoint.

Live visualization in the browser using Chart.js.

Real-Time EEG Graphs

Web dashboard (/graph) shows all 6 channels updating live.

Useful for monitoring signal quality and electrode placement.

Color Prediction via ML
Two modes of operation:

SSVEP Mode (Recommended):

Colored tiles flicker at unique frequencies (8–20 Hz).

The user stares at a color → brain’s visual cortex locks to that frequency.

FFT analysis detects the frequency → maps to the intended color.

Imagined Color Mode (Experimental):

User imagines a color without visual stimulus.

Bandpower features are extracted from EEG signals.

Trained LDA/SVM model predicts the intended color.

Python ML Backend

Flask server fetches EEG from ESP32.

Performs FFT-based frequency decoding (SSVEP) or feature extraction + classifier prediction (Imagined).

Exposes /predict_ssvep and /predict_imagined endpoints for real-time results.

Frontend Interfaces

EEG Graph UI → live signal chart.

SSVEP Stimulus UI → flickering colored tiles for attention-based selection.

Prediction UI → displays the detected color with confidence level.

🛠️ Tech Stack

Hardware: ESP32, BioAmp / EEG frontend, 6-channel electrode setup.

Firmware: C++ (Arduino), Wi-Fi WebServer for streaming data.

Backend: Python (Flask, NumPy, Scikit-Learn).

Frontend: HTML, Chart.js, Vanilla JS.

🚀 How It Works

ESP32 collects EEG → streams raw ADC values via Wi-Fi.

Python backend subscribes to EEG stream and buffers signals.

Signal Processing & ML

SSVEP: FFT finds the strongest frequency component → maps to color.

Imagined: Bandpower features extracted → passed to trained classifier.

Web UI shows prediction → background color updates in real time.

📊 Example Demo

Live EEG graph from 6 channels.

Flickering colored tiles → user stares at one → predicted color updates.

Optional: train custom models for imagined color classification.

⚠️ Disclaimer

This is a prototype research/demo project, not a medical device. EEG is very sensitive to noise and requires proper electrode placement and signal conditioning.
