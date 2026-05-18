/*
 * BioAmp EXG Pill — Dual-Channel EEG  (ML-ready edition)
 * ========================================================
 *
 * Pill 1 — Frontal channel  (GPIO 34 / ADC1_CH6)
 *   IN+  → Fp1  (left forehead)
 *   IN-  → Fp2  (right forehead)
 *   REF  → M1   (left mastoid — hard bump behind left earlobe)
 *   VCC  → 3.3 V
 *   GND  → GND
 *
 * Pill 2 — Temporal channel  (GPIO 35 / ADC1_CH7)
 *   IN+  → T7   (above left ear)
 *   IN-  → T8   (above right ear)
 *   REF  → M2   (right mastoid — hard bump behind right earlobe)
 *   VCC  → 3.3 V  (shared rail with Pill 1)
 *   GND  → GND    (shared rail with Pill 1)
 *
 * WebSocket JSON (port 81) — one packet per 1-second epoch:
 *   ts, epoch, sid, quality, quality2
 *   filt_rms, filt_rms2, raw_rms, hum_pct
 *   ab_ratio, ta_ratio, dom
 *   ch[]   — 256 int16 samples × 10  (Ch1, divide by 10 for µV)
 *   ch2[]  — 256 int16 samples × 10  (Ch2, divide by 10 for µV)
 *   bp     — {d,t,a,b,g}  Ch1 absolute band power
 *   bp2    — {d,t,a,b,g}  Ch2 absolute band power
 *   rel    — {d,t,a,b,g}  Ch1 relative band power
 *   rel2   — {d,t,a,b,g}  Ch2 relative band power
 *
 * Serial output (115200 baud) — one line per epoch:
 *   Ch1: D,T,A,B,G rel | AB,TA ratios | Frms,Rrms,Hum | Q | Ep,Ms | abs powers
 *   Ch2: D2,T2,A2,B2,G2 rel | AB2,TA2 | Frms2,Q2
 *
 * Libraries (Tools → Manage Libraries):
 *   WebSockets  by Markus Sattler
 *   ArduinoJson by Benoit Blanchon
 */

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <math.h>
#include "html_page.h"

// ── WiFi ──────────────────────────────────────────────────────────────────────
const char* SSID     = "HIVE 5TH FLOOR A";
const char* PASSWORD = "raghav123";

// ── Static IP ─────────────────────────────────────────────────────────────────
IPAddress LOCAL_IP(192, 168,   1,  33);
IPAddress GATEWAY (192, 168,   1,   1);
IPAddress SUBNET  (255, 255, 255,   0);
IPAddress DNS_SRV (  8,   8,   8,   8);

// ── ADC pins ──────────────────────────────────────────────────────────────────
#define EEG_PIN1    34          // ADC1_CH6 — Pill 1: Frontal  Fp1-Fp2
#define EEG_PIN2    35          // ADC1_CH7 — Pill 2: Temporal T7-T8
#define ADC_BITS    12
#define ADC_MAXV    4095.0f
#define V_SUPPLY    3.3f
#define V_BIAS      1.65f       // EXG Pill mid-rail
#define AMP_GAIN    1000.0f     // BioAmp EXG Pill gain
#define OVERSAMPLE  8           // 8x oversample → noise floor / ~3x

// ── Sampling ──────────────────────────────────────────────────────────────────
#define FS   256                // Hz
#define N    256                // samples per epoch = 1 s, 1 Hz FFT resolution

// ── Biquad IIR ────────────────────────────────────────────────────────────────
struct Biquad {
  float b0, b1, b2, a1, a2;
  float x1=0, x2=0, y1=0, y2=0;
  float run(float x) {
    float y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
    x2=x1; x1=x; y2=y1; y1=y; return y;
  }
};

// Fs = 256 Hz coefficients
// High-pass 0.5 Hz
Biquad hpf1  = {0.99844741f,-1.99689482f,0.99844741f,-1.99689142f, 0.99689822f};
Biquad hpf2  = {0.99844741f,-1.99689482f,0.99844741f,-1.99689142f, 0.99689822f};

// Low-pass 45 Hz
Biquad lpf1  = {0.09763107f, 0.19526215f,0.09763107f,-0.94280904f, 0.33483138f};
Biquad lpf2  = {0.09763107f, 0.19526215f,0.09763107f,-0.94280904f, 0.33483138f};

// Notch 50 Hz (Q=30)
Biquad ntch1 = {0.98208681f,-1.29609948f,0.98208681f,-1.29609948f, 0.96417362f};
Biquad ntch2 = {0.98208681f,-1.29609948f,0.98208681f,-1.29609948f, 0.96417362f};

// ── Sample buffers ────────────────────────────────────────────────────────────
int16_t buf1[N], buf2[N];       // filtered (µV × 10) — both channels
float raw1[N],   raw2[N];       // pre-filter µV      — both channels
int     bufIdx = 0;
uint32_t epochNo = 0;

// ── Network ───────────────────────────────────────────────────────────────────
WiFiServer       httpSrv(80);
WebSocketsServer wsSrv(81);
bool wifiOk = false;

unsigned long nextSampleUs   = 0;
unsigned long sessionStartMs = 0;
String        sessionId;

// ── Helpers ───────────────────────────────────────────────────────────────────
static inline float adcToUv(int acc) {
  float vadc = ((float)acc / (OVERSAMPLE * ADC_MAXV)) * V_SUPPLY;
  return ((vadc - V_BIAS) / AMP_GAIN) * 1.0e6f;
}

// Real radix-2 FFT (in-place, N must be power of 2)
static void fft(float* x, int sz) {
  for (int i=1,j=0; i<sz; i++) {
    int bit=sz>>1;
    for (; j&bit; bit>>=1) j^=bit;
    j^=bit;
    if (i<j) { float t=x[i]; x[i]=x[j]; x[j]=t; }
  }
  for (int len=2; len<=sz; len<<=1) {
    float ang=-2.0f*(float)M_PI/(float)len;
    float wr=cosf(ang), wi=sinf(ang);
    for (int i=0; i<sz; i+=len) {
      float ur=1.0f, ui=0.0f;
      for (int k=0; k<len/2; k++) {
        float tr=ur*x[i+k+len/2];
        x[i+k+len/2]=x[i+k]-tr; x[i+k]+=tr;
        float nu=ur*wr-ui*wi; ui=ur*wi+ui*wr; ur=nu;
      }
    }
  }
}

static float bandPow(float* m, int lo, int hi, int half) {
  float p=0;
  for (int k=lo; k<=hi && k<half; k++) p+=m[k]*m[k];
  return p;
}

// Per-channel analysis: fills bp[5] (abs), rel[5], abRatio, taRatio, filtRms
static void analyseChannel(int16_t* buf, float* bpOut, float* relOut,
                            float& abRatio, float& taRatio, float& filtRms) {
  float w[N];
  for (int s=0; s<N; s++) {
    float h = 0.5f*(1.0f-cosf(2.0f*(float)M_PI*s/(N-1)));
    w[s] = (buf[s]/10.0f)*h;
  }
  fft(w, N);

  bpOut[0] = bandPow(w,  1,  4, N/2);  // Delta  1-4 Hz
  bpOut[1] = bandPow(w,  4,  8, N/2);  // Theta  4-8 Hz
  bpOut[2] = bandPow(w,  8, 13, N/2);  // Alpha  8-13 Hz
  bpOut[3] = bandPow(w, 13, 30, N/2);  // Beta  13-30 Hz
  bpOut[4] = bandPow(w, 30, 45, N/2);  // Gamma 30-45 Hz

  float total = bpOut[0]+bpOut[1]+bpOut[2]+bpOut[3]+bpOut[4];
  if (total < 1e-6f) total=1e-6f;
  for (int i=0; i<5; i++) relOut[i] = bpOut[i]/total;

  abRatio = (bpOut[3]>1e-6f) ? bpOut[2]/bpOut[3] : 0.0f;
  taRatio = (bpOut[2]>1e-6f) ? bpOut[1]/bpOut[2] : 0.0f;

  float sumSq=0;
  for (int s=0; s<N; s++) { float v=buf[s]/10.0f; sumSq+=v*v; }
  filtRms = sqrtf(sumSq/N);
}

// Quality string from raw channel buffer
static const char* channelQuality(float* rawBuf, float filtRms) {
  float rawWork[N];
  for (int s=0; s<N; s++) {
    float h=0.5f*(1.0f-cosf(2.0f*(float)M_PI*s/(N-1)));
    rawWork[s]=rawBuf[s]*h;
  }
  fft(rawWork, N);
  float hum50=0, totalPow=0;
  for (int k=48; k<=52; k++) hum50  += rawWork[k]*rawWork[k];
  for (int k=1;  k<N/2;  k++) totalPow += rawWork[k]*rawWork[k];
  float humRatio = (totalPow>0)?(hum50/totalPow):0;
  if (humRatio>0.35f) return "floating";
  if (filtRms  <0.8f) return "flat";
  return "good";
}

// ── HTTP ──────────────────────────────────────────────────────────────────────
void handleHTTP() {
  WiFiClient client=httpSrv.available();
  if (!client) return;
  unsigned long t0=millis();
  while (!client.available() && millis()-t0<1500) delay(1);
  while (client.available()) client.read();
  client.print(F("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n"));
  client.print(FPSTR(HTML));
  client.stop();
}
void onWSEvent(uint8_t,WStype_t,uint8_t*,size_t){}

// ── setup() ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println("\n=== BioAmp EXG Pill x2 - Dual-Channel EEG ===");
  Serial.println("  CH1 (GPIO34): Fp1(IN+) Fp2(IN-) M1(REF) — Frontal");
  Serial.println("  CH2 (GPIO35): T7 (IN+) T8 (IN-) M2(REF) — Temporal");
  Serial.println("FORMAT: Ch1 D,T,A,B,G|AB,TA|Frms,Rrms,Hum,Q  Ch2 D2,T2,A2,B2,G2|AB2,TA2|Frms2,Q2");

  analogReadResolution(ADC_BITS);
  analogSetAttenuation(ADC_11db);

  // ── Static IP — mode MUST be set before config() on ESP32 ────────────────
  WiFi.persistent(false);          // don't write config to flash every boot
  WiFi.mode(WIFI_STA);             // station mode first — required for config()
  WiFi.setAutoReconnect(true);     // auto-reconnect if AP drops

  Serial.printf("WiFi: reserving 192.168.1.33, connecting to \"%s\"...\n", SSID);
  if (!WiFi.config(LOCAL_IP, GATEWAY, SUBNET, DNS_SRV))
    Serial.println("WiFi: WARNING — static IP config rejected, will use DHCP");

  WiFi.begin(SSID, PASSWORD);
  int tries=0;
  while (WiFi.status()!=WL_CONNECTED && tries<30) { delay(400); tries++; }

  if (WiFi.status()==WL_CONNECTED) {
    wifiOk=true;
    IPAddress got = WiFi.localIP();
    Serial.printf("WiFi: connected  IP=%s\n", got.toString().c_str());

    // Warn loudly if router gave a different IP than requested
    if (got != LOCAL_IP) {
      Serial.printf("WiFi: WARNING — got %s instead of 192.168.1.33\n",
                    got.toString().c_str());
      Serial.println("       Add a DHCP reservation in your router for this MAC:");
      uint8_t mac[6]; WiFi.macAddress(mac);
      Serial.printf("       MAC = %02X:%02X:%02X:%02X:%02X:%02X\n",
                    mac[0],mac[1],mac[2],mac[3],mac[4],mac[5]);
    }

    uint8_t mac[6]; WiFi.macAddress(mac);
    sessionId = String(mac[4],HEX)+String(mac[5],HEX)+String(millis()&0xFFFF,HEX);
    httpSrv.begin(); wsSrv.begin(); wsSrv.onEvent(onWSEvent);

    Serial.println("─────────────────────────────────────────");
    Serial.printf("  Dashboard : http://192.168.1.33\n");
    Serial.printf("  WebSocket : ws://192.168.1.33:81\n");
    Serial.println("─────────────────────────────────────────");
  } else {
    wifiOk=false;
    Serial.println("WiFi: not connected — running serial-only mode");
  }

  Serial.println("READY — dual-channel EEG streaming");
  delay(500);
  sessionStartMs=millis();
  nextSampleUs  =micros();
}

// ── loop() ────────────────────────────────────────────────────────────────────
void loop() {
  if (wifiOk) { wsSrv.loop(); handleHTTP(); }

  if ((long)(micros()-nextSampleUs)<0) return;
  nextSampleUs += (1000000UL/FS);

  // ── Sample both channels (interleaved for near-simultaneity) ─────────────
  int acc1=0, acc2=0;
  for (int k=0; k<OVERSAMPLE; k++) {
    acc1 += analogRead(EEG_PIN1);
    acc2 += analogRead(EEG_PIN2);
  }
  float uVraw1 = adcToUv(acc1);
  float uVraw2 = adcToUv(acc2);

  // ── Per-channel filter chains ─────────────────────────────────────────────
  float uV1 = ntch1.run(lpf1.run(hpf1.run(uVraw1)));
  float uV2 = ntch2.run(lpf2.run(hpf2.run(uVraw2)));

  raw1[bufIdx] = uVraw1;   raw2[bufIdx] = uVraw2;
  buf1[bufIdx] = (int16_t)constrain((int)(uV1*10.0f),-32768,32767);
  buf2[bufIdx] = (int16_t)constrain((int)(uV2*10.0f),-32768,32767);
  bufIdx++;

  if (bufIdx < N) return;
  bufIdx=0; epochNo++;

  // ── Raw RMS (Ch1 — used for overall session quality) ─────────────────────
  float rawSumSq=0;
  for (int s=0; s<N; s++) rawSumSq+=raw1[s]*raw1[s];
  float rawRms=sqrtf(rawSumSq/N);

  // ── Analyse both channels ─────────────────────────────────────────────────
  float bp1[5], rel1[5], ab1, ta1, frms1;
  float bp2[5], rel2[5], ab2, ta2, frms2;
  analyseChannel(buf1, bp1, rel1, ab1, ta1, frms1);
  analyseChannel(buf2, bp2, rel2, ab2, ta2, frms2);

  const char* q1 = channelQuality(raw1, frms1);
  const char* q2 = channelQuality(raw2, frms2);

  // Hum ratio for serial (Ch1)
  float rawWork[N];
  for (int s=0; s<N; s++) {
    float h=0.5f*(1.0f-cosf(2.0f*(float)M_PI*s/(N-1)));
    rawWork[s]=raw1[s]*h;
  }
  fft(rawWork, N);
  float hum50=0, totalPow=0;
  for (int k=48; k<=52; k++) hum50   +=rawWork[k]*rawWork[k];
  for (int k=1;  k<N/2;  k++) totalPow+=rawWork[k]*rawWork[k];
  float humPct=(totalPow>0)?(hum50/totalPow*100.0f):0;

  // Dominant band (Ch1)
  const char* bnames[5]={"DELTA","THETA","ALPHA","BETA","GAMMA"};
  int dom=0;
  for (int i=1; i<5; i++) if (bp1[i]>bp1[dom]) dom=i;

  // ── Serial — Ch1 then Ch2 on the same line ─────────────────────────────────
  int qc1=(strcmp(q1,"good")==0)?2:(strcmp(q1,"floating")==0)?1:0;
  int qc2=(strcmp(q2,"good")==0)?2:(strcmp(q2,"floating")==0)?1:0;
  Serial.printf(
    "D:%.4f,T:%.4f,A:%.4f,B:%.4f,G:%.4f,"
    "AB:%.3f,TA:%.3f,Frms:%.1f,Rrms:%.1f,Hum:%.1f,Q:%d,"
    "Ep:%lu,Ms:%lu,"
    "Da:%.3f,Ta:%.3f,Aa:%.3f,Ba:%.3f,Ga:%.3f,"
    "D2:%.4f,T2:%.4f,A2:%.4f,B2:%.4f,G2:%.4f,"
    "AB2:%.3f,TA2:%.3f,Frms2:%.1f,Q2:%d\n",
    rel1[0],rel1[1],rel1[2],rel1[3],rel1[4],
    ab1,ta1,frms1,rawRms,humPct,qc1,
    epochNo,(unsigned long)(millis()-sessionStartMs),
    bp1[0],bp1[1],bp1[2],bp1[3],bp1[4],
    rel2[0],rel2[1],rel2[2],rel2[3],rel2[4],
    ab2,ta2,frms2,qc2
  );

  // ── WebSocket JSON ────────────────────────────────────────────────────────
  if (wifiOk && wsSrv.connectedClients()>0) {
    // 2× ch arrays (256 samples each ≈ 1.8 KB) + metadata ≈ 5 KB total
    DynamicJsonDocument doc(6000);

    doc["ts"]        = (uint32_t)(millis()-sessionStartMs);
    doc["epoch"]     = epochNo;
    doc["sid"]       = sessionId;
    // Ch1 quality / metrics
    doc["quality"]   = q1;
    doc["filt_rms"]  = roundf(frms1*10.0f)/10.0f;
    doc["raw_rms"]   = roundf(rawRms*10.0f)/10.0f;
    doc["hum_pct"]   = roundf(humPct*10.0f)/10.0f;
    doc["ab_ratio"]  = roundf(ab1*100.0f)/100.0f;
    doc["ta_ratio"]  = roundf(ta1*100.0f)/100.0f;
    doc["dom"]       = bnames[dom];
    // Ch2 quality / metrics
    doc["quality2"]  = q2;
    doc["filt_rms2"] = roundf(frms2*10.0f)/10.0f;
    doc["ab_ratio2"] = roundf(ab2*100.0f)/100.0f;
    doc["ta_ratio2"] = roundf(ta2*100.0f)/100.0f;

    // Raw sample arrays — int16 × 10 (divide by 10 for µV)
    JsonArray ch1arr = doc.createNestedArray("ch");
    for (int s=0; s<N; s++) ch1arr.add(buf1[s]);

    JsonArray ch2arr = doc.createNestedArray("ch2");
    for (int s=0; s<N; s++) ch2arr.add(buf2[s]);

    // Ch1 band powers
    JsonObject bpObj = doc.createNestedObject("bp");
    bpObj["d"]=roundf(bp1[0]*10.0f)/10.0f; bpObj["t"]=roundf(bp1[1]*10.0f)/10.0f;
    bpObj["a"]=roundf(bp1[2]*10.0f)/10.0f; bpObj["b"]=roundf(bp1[3]*10.0f)/10.0f;
    bpObj["g"]=roundf(bp1[4]*10.0f)/10.0f;

    // Ch2 band powers
    JsonObject bp2Obj = doc.createNestedObject("bp2");
    bp2Obj["d"]=roundf(bp2[0]*10.0f)/10.0f; bp2Obj["t"]=roundf(bp2[1]*10.0f)/10.0f;
    bp2Obj["a"]=roundf(bp2[2]*10.0f)/10.0f; bp2Obj["b"]=roundf(bp2[3]*10.0f)/10.0f;
    bp2Obj["g"]=roundf(bp2[4]*10.0f)/10.0f;

    // Ch1 relative powers
    JsonObject relObj = doc.createNestedObject("rel");
    relObj["d"]=roundf(rel1[0]*1000.0f)/1000.0f; relObj["t"]=roundf(rel1[1]*1000.0f)/1000.0f;
    relObj["a"]=roundf(rel1[2]*1000.0f)/1000.0f; relObj["b"]=roundf(rel1[3]*1000.0f)/1000.0f;
    relObj["g"]=roundf(rel1[4]*1000.0f)/1000.0f;

    // Ch2 relative powers
    JsonObject rel2Obj = doc.createNestedObject("rel2");
    rel2Obj["d"]=roundf(rel2[0]*1000.0f)/1000.0f; rel2Obj["t"]=roundf(rel2[1]*1000.0f)/1000.0f;
    rel2Obj["a"]=roundf(rel2[2]*1000.0f)/1000.0f; rel2Obj["b"]=roundf(rel2[3]*1000.0f)/1000.0f;
    rel2Obj["g"]=roundf(rel2[4]*1000.0f)/1000.0f;

    String json; json.reserve(5500);
    serializeJson(doc, json);
    wsSrv.broadcastTXT(json);
  }
}
