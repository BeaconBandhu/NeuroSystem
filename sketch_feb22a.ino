#define PIEZO_PIN 34  // Analog input pin for piezo sensor

// Thresholds
int heartThreshold = 240;
int breathThreshold = 100;

// Heart Rate Variables
unsigned long lastBeatTime = 0;
float bpm = 0;
bool pulseDetected = false;

// Respiration Rate Variables
unsigned long lastBreathTime = 0;
float rr = 0;
bool inhaleDetected = false;

// HRV & Stress Variables
float rrIntervals[10];
int rrIndex = 0;
float rmssd = 0;
float stressLevel = 0;

// Fatigue Index Variables
unsigned long startTime, recoveryTime;
bool startFlag = false;
float initialHR = 0, recoveryHR = 0, FI = 0;

void setup() {
    Serial.begin(115200);
}

void loop() {
    int sensorValue = analogRead(PIEZO_PIN);

    // ---- HEART RATE (BPM) DETECTION ----
    if (sensorValue > heartThreshold && !pulseDetected) {
        unsigned long currentTime = millis();
        if (currentTime - lastBeatTime > 600) {
            float rrInterval = (currentTime - lastBeatTime) / 1000.0;
            bpm = 60.0 / rrInterval;
            if (bpm > 40 && bpm < 180) {
                rrIntervals[rrIndex] = rrInterval;
                rrIndex = (rrIndex + 1) % 10;
            }
            lastBeatTime = currentTime;
            pulseDetected = true;
        }
    } else if (sensorValue < (heartThreshold - 30)) {
        pulseDetected = false;
    }

    // ---- RESPIRATION RATE (RR) DETECTION ----
    if (sensorValue > breathThreshold && !inhaleDetected) {
        unsigned long currentTime = millis();
        if (currentTime - lastBreathTime > 3000) {
            rr = 60.0 / ((currentTime - lastBreathTime) / 1000.0);
            if (rr > 8 && rr < 30) {
                lastBreathTime = currentTime;
            }
            inhaleDetected = true;
        }
    } else if (sensorValue < (breathThreshold - 20)) {
        inhaleDetected = false;
    }

    // ---- FATIGUE INDEX (FI) CALCULATION ----
    if (!startFlag && bpm > 80) {
        initialHR = bpm;
        startTime = millis();
        startFlag = true;
    }

    if (startFlag && millis() - startTime >= 30000) {
        recoveryHR = bpm;
        recoveryTime = (millis() - startTime) / 1000;
        FI = (initialHR - recoveryHR) / recoveryTime;
        startFlag = false;
    }

    // ---- HRV (RMSSD) & STRESS LEVEL CALCULATION ----
    if (rrIndex >= 9) {
        float sumSquaredDiffs = 0;
        for (int i = 0; i < 9; i++) {
            float diff = rrIntervals[i + 1] - rrIntervals[i];
            sumSquaredDiffs += diff * diff;
        }
        rmssd = sqrt(sumSquaredDiffs / 9.0);
        stressLevel = map(rmssd * 1000, 10, 150, 100, 0);
        stressLevel = constrain(stressLevel, 0, 100);
    }

    // ---- SERIAL COMMUNICATION ----
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        if (command == "GET_DATA") {
            Serial.print(bpm);
            Serial.print(",");
            Serial.print(rr);
            Serial.print(",");
            Serial.print(FI);
            Serial.print(",");
            Serial.print(stressLevel);
            Serial.print(",");
            Serial.println(120);  // Placeholder BP value
        }
    }
    
    delay(10);
}
