/*
 * html_page.h — kept separate so Arduino's prototype-scanner never sees
 * the JavaScript 'function' keyword and mistakes it for C++ code.
 */
#pragma once
#include <pgmspace.h>

static const char HTML[] PROGMEM = R"EEG_PAGE(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG Monitor — Dual Channel</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',Arial,sans-serif;font-size:14px}
header{background:#161b22;padding:12px 20px;border-bottom:1px solid #30363d;
       display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
header h1{font-size:1rem;color:#58a6ff}
.hinfo{display:flex;gap:16px;font-size:.76rem;color:#8b949e;align-items:center;flex-wrap:wrap}
.dot{width:9px;height:9px;border-radius:50%;background:#f85149;
     display:inline-block;margin-right:5px;vertical-align:middle}
.dot.live{background:#3fb950;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
main{padding:16px;display:flex;flex-direction:column;gap:16px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card h2{font-size:.82rem;color:#58a6ff;text-transform:uppercase;
          letter-spacing:.5px;margin-bottom:10px}
.card h2 .ch-tag{font-size:.72rem;padding:2px 8px;border-radius:10px;
                  margin-left:8px;font-weight:600;vertical-align:middle}
.ch1-tag{background:#0d2318;color:#3fb950;border:1px solid #3fb950}
.ch2-tag{background:#0d1e35;color:#58a6ff;border:1px solid #58a6ff}
.chart-wrap{position:relative;height:180px}
.bands{display:flex;gap:6px;margin-top:12px;flex-wrap:wrap}
.band{flex:1;min-width:80px;background:#0d1117;border-radius:6px;padding:8px 10px}
.band .bn{font-size:.68rem;color:#8b949e;margin-bottom:3px}
.band .bv{font-size:.95rem;font-weight:700;color:#e6edf3;margin-bottom:4px}
.band .bar{height:5px;border-radius:3px;width:0%;transition:width .5s ease}
.dominant{border:1px solid #ffa657!important}
.dominant .bn{color:#ffa657!important}
.info-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.info-pill{background:#0d1117;border-radius:20px;padding:4px 12px;
           font-size:.72rem;color:#8b949e;border:1px solid #30363d}
.footer{padding:14px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;
        background:#161b22;border-top:1px solid #30363d}
button{padding:8px 18px;border-radius:6px;border:none;cursor:pointer;
       font-size:.82rem;font-weight:600;transition:opacity .15s}
button:hover{opacity:.8}
#btnStart{background:#238636;color:#fff}
#btnPause{background:#e3b341;color:#000}
#btnResume{background:#1f6feb;color:#fff}
#btnStop{background:#da3633;color:#fff}
#btnDL{background:#30363d;color:#e6edf3}
#btnClr{background:#21262d;color:#8b949e}
#finfo{font-size:.72rem;color:#8b949e}
.state-badge{font-size:.72rem;font-weight:600;padding:3px 10px;border-radius:12px;
             background:#0d1117;border:1px solid #30363d;color:#8b949e}
.state-badge.recording{background:#0d2318;border-color:#3fb950;color:#3fb950}
.state-badge.paused{background:#1e1a0e;border-color:#e3b341;color:#e3b341}
.quality-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:4px}
.quality-bar{display:flex;align-items:center;gap:10px;padding:9px 14px;
             border-radius:7px;font-size:.80rem;font-weight:600;
             border:1px solid #30363d;background:#0d1117;transition:all .3s}
.quality-bar.good{background:#0d2318;border-color:#3fb950;color:#3fb950}
.quality-bar.flat{background:#1e1a0e;border-color:#e3b341;color:#e3b341}
.quality-bar.floating{background:#1e1a0e;border-color:#e3b341;color:#e3b341}
.quality-bar.disconnected{background:#2d0f10;border-color:#f85149;color:#f85149}
.ql-icon{font-size:1.1rem;flex-shrink:0}
.ql-metrics{margin-left:auto;display:flex;gap:10px;font-size:.68rem;color:#8b949e;font-weight:400;flex-wrap:wrap}
</style>
</head>
<body>
<header>
  <h1>&#x26A1; EEG Monitor &mdash; BioAmp EXG Pill &times; 2 &mdash; Dual Channel</h1>
  <div class="hinfo">
    <span><span class="dot" id="dot"></span><span id="connTxt">Connecting...</span></span>
    <span id="ipTxt"></span>
    <span id="smplTxt">0 epochs</span>
    <span id="timeTxt">00:00</span>
  </div>
</header>
<main>

  <!-- Signal quality banners -->
  <div class="quality-row">
    <div class="quality-bar" id="qualBar1">
      <span class="ql-icon" id="qlIcon1">&#9711;</span>
      <span id="qlText1">CH1 &mdash; Waiting...</span>
      <div class="ql-metrics">
        <span>Raw: <b id="qlRaw1">--</b> uV</span>
        <span>Filt: <b id="qlFilt1">--</b> uV</span>
        <span>Hum: <b id="qlHum1">--</b>%</span>
        <span>A/B: <b id="qlAB1">--</b></span>
        <span>T/A: <b id="qlTA1">--</b></span>
      </div>
    </div>
    <div class="quality-bar" id="qualBar2">
      <span class="ql-icon" id="qlIcon2">&#9711;</span>
      <span id="qlText2">CH2 &mdash; Waiting...</span>
      <div class="ql-metrics">
        <span>Filt: <b id="qlFilt2">--</b> uV</span>
        <span>A/B: <b id="qlAB2">--</b></span>
        <span>T/A: <b id="qlTA2">--</b></span>
      </div>
    </div>
  </div>

  <!-- Electrode setup -->
  <div class="card">
    <h2>Electrode Setup</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:.76rem;color:#8b949e">
      <div>
        <div style="color:#3fb950;font-weight:600;margin-bottom:6px">CH1 — Frontal (GPIO 34)</div>
        IN+ &rarr; Fp1 (left forehead) &nbsp;|&nbsp;
        IN&minus; &rarr; Fp2 (right forehead) &nbsp;|&nbsp;
        REF &rarr; M1 (left mastoid)
      </div>
      <div>
        <div style="color:#58a6ff;font-weight:600;margin-bottom:6px">CH2 — Temporal (GPIO 35)</div>
        IN+ &rarr; T7 (above left ear) &nbsp;|&nbsp;
        IN&minus; &rarr; T8 (above right ear) &nbsp;|&nbsp;
        REF &rarr; M2 (right mastoid)
      </div>
    </div>
    <div class="info-row" style="margin-top:10px">
      <span class="info-pill">256 Hz sampling</span>
      <span class="info-pill">1 s epoch</span>
      <span class="info-pill">1 Hz FFT resolution</span>
      <span class="info-pill">HP 0.5 Hz &middot; LP 45 Hz &middot; Notch 50 Hz</span>
    </div>
  </div>

  <!-- Live EEG waveforms -->
  <div class="row2">
    <div class="card">
      <h2>Live EEG <span class="ch-tag ch1-tag">CH1 Frontal</span></h2>
      <div class="chart-wrap"><canvas id="cvEEG1"></canvas></div>
    </div>
    <div class="card">
      <h2>Live EEG <span class="ch-tag ch2-tag">CH2 Temporal</span></h2>
      <div class="chart-wrap"><canvas id="cvEEG2"></canvas></div>
    </div>
  </div>

  <!-- Band power -->
  <div class="row2">
    <div class="card">
      <h2>Band Power <span class="ch-tag ch1-tag">CH1 Frontal</span></h2>
      <div class="bands" id="bands1"></div>
    </div>
    <div class="card">
      <h2>Band Power <span class="ch-tag ch2-tag">CH2 Temporal</span></h2>
      <div class="bands" id="bands2"></div>
    </div>
  </div>

</main>
<div class="footer">
  <button id="btnStart"  onclick="recStart()">&#9654; Start Recording</button>
  <button id="btnPause"  onclick="recPause()"  style="display:none">&#9646;&#9646; Pause</button>
  <button id="btnResume" onclick="recResume()" style="display:none">&#9654; Resume</button>
  <button id="btnStop"   onclick="recStop()"   style="display:none">&#9632; Stop</button>
  <button id="btnDL"     onclick="dlJSON()">&#11015; Download JSON</button>
  <button id="btnClr"    onclick="clearSession()">&#10005; Clear</button>
  <span class="state-badge idle" id="stateBadge">Idle</span>
  <span id="finfo">Press Start Recording to begin.</span>
</div>

<script>
var BAND_KEYS  = ['d','t','a','b','g'];
var BAND_NAMES = ['Delta','Theta','Alpha','Beta','Gamma'];
var BAND_RANGE = ['0.5-4 Hz','4-8 Hz','8-13 Hz','13-30 Hz','30-45 Hz'];
var BAND_COLS  = ['#58a6ff','#3fb950','#ffa657','#d2a8ff','#f78166'];

var WIN   = 300;
var DECIM = 4;

var dispQ1 = [], dispQ2 = [];
var allBatches = [], totalEpochs = 0;
var sessStart  = null;

var QUALITY_CFG = {
  good:         { icon: '&#9679;', text: 'Good — skin contact confirmed' },
  flat:         { icon: '&#9644;', text: 'Flat — electrode not seated' },
  floating:     { icon: '&#9888;', text: 'Floating — 50 Hz dominant' },
  disconnected: { icon: '&#9888;', text: 'Disconnected' }
};

// Build band cards for both channels
function buildBands(containerId, prefix) {
  var div = document.getElementById(containerId);
  var elems = {};
  for (var bi = 0; bi < BAND_KEYS.length; bi++) {
    var k = BAND_KEYS[bi];
    var d = document.createElement('div');
    d.className = 'band'; d.id = prefix + k;
    d.innerHTML =
      '<div class="bn">' + BAND_NAMES[bi] +
      ' <span style="font-size:.63rem">(' + BAND_RANGE[bi] + ')</span></div>' +
      '<div class="bv" id="' + prefix + 'v_' + k + '">--</div>' +
      '<div class="bar" id="' + prefix + 'b_' + k + '" style="background:' + BAND_COLS[bi] + '"></div>';
    div.appendChild(d);
    elems[k] = {
      val:  document.getElementById(prefix + 'v_' + k),
      bar:  document.getElementById(prefix + 'b_' + k),
      card: d
    };
  }
  return elems;
}
var be1 = buildBands('bands1', 'c1_');
var be2 = buildBands('bands2', 'c2_');

// Chart factory
function makeChart(canvasId, color) {
  var empty = [];
  for (var i = 0; i < WIN; i++) empty.push(null);
  var ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: empty.slice(),
      datasets: [{
        data: empty.slice(),
        borderColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15,
        fill: false
      }]
    },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: {
          grid: { color: '#21262d' },
          ticks: { color: '#8b949e', font: { size: 10 }, maxTicksLimit: 6 },
          title: { display: true, text: 'uV', color: '#8b949e', font: { size: 10 } }
        }
      }
    }
  });
}
var chart1 = makeChart('cvEEG1', '#3fb950');
var chart2 = makeChart('cvEEG2', '#58a6ff');

// Drain display queues at ~32 Hz
setInterval(function() {
  function drain(q, chart) {
    if (!q.length) return;
    var n = Math.min(2, q.length);
    var ds = chart.data.datasets[0].data;
    for (var k = 0; k < n; k++) {
      ds.push(q.shift());
      if (ds.length > WIN) ds.shift();
    }
    var lbls = [];
    for (var i = 0; i < ds.length; i++) lbls.push('');
    chart.data.labels = lbls;
    chart.update('none');
  }
  drain(dispQ1, chart1);
  drain(dispQ2, chart2);
}, 31);

// Recording state machine
var recState = 'idle';
function setRecState(s) {
  recState = s;
  var badge = document.getElementById('stateBadge');
  document.getElementById('btnStart').style.display  = (s === 'idle')      ? '' : 'none';
  document.getElementById('btnPause').style.display  = (s === 'recording') ? '' : 'none';
  document.getElementById('btnResume').style.display = (s === 'paused')    ? '' : 'none';
  document.getElementById('btnStop').style.display   = (s !== 'idle')      ? '' : 'none';
  badge.className = 'state-badge ' + s;
  badge.textContent = s.charAt(0).toUpperCase() + s.slice(1);
  var fi = document.getElementById('finfo');
  if (s === 'idle')      fi.textContent = 'Press Start Recording to begin.';
  if (s === 'recording') fi.textContent = 'Recording...';
  if (s === 'paused')    fi.textContent = 'Paused — data not being saved.';
}
function recStart() {
  allBatches = []; totalEpochs = 0;
  document.getElementById('smplTxt').textContent = '0 epochs';
  setRecState('recording');
  if (!ws || ws.readyState !== WebSocket.OPEN) connect();
}
function recPause()  { setRecState('paused'); }
function recResume() { setRecState('recording'); }
function recStop() {
  setRecState('idle');
  document.getElementById('finfo').textContent =
    'Stopped. ' + allBatches.length + ' epochs — click Download JSON to export.';
}

// WebSocket — hardcoded to reserved IP 192.168.1.33:81
var WS_URL = 'ws://192.168.1.33:81';
var ws, reconnT;
function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = function() {
    clearTimeout(reconnT);
    document.getElementById('dot').classList.add('live');
    document.getElementById('connTxt').textContent = 'Live';
    document.getElementById('ipTxt').textContent = '@ 192.168.1.33:81';
    if (!sessStart) {
      sessStart = Date.now();
      setInterval(function() {
        var s  = Math.floor((Date.now() - sessStart) / 1000);
        var m  = Math.floor(s / 60); var ss = s % 60;
        document.getElementById('timeTxt').textContent =
          (m < 10 ? '0' : '') + m + ':' + (ss < 10 ? '0' : '') + ss;
      }, 1000);
    }
  };
  ws.onclose = function() {
    document.getElementById('dot').classList.remove('live');
    document.getElementById('connTxt').textContent = 'Reconnecting...';
    reconnT = setTimeout(connect, 2500);
  };
  ws.onmessage = function(e) {
    var msg; try { msg = JSON.parse(e.data); } catch(err) { return; }
    onBatch(msg);
  };
}

function updateQuality(id, iconId, textId, q, label, metrics) {
  var bar = document.getElementById(id);
  var cfg = QUALITY_CFG[q] || QUALITY_CFG['flat'];
  bar.className = 'quality-bar ' + q;
  document.getElementById(iconId).innerHTML   = cfg.icon;
  document.getElementById(textId).textContent = label + ' — ' + cfg.text;
  for (var k in metrics) {
    var el = document.getElementById(k);
    if (el) el.textContent = metrics[k];
  }
}

function updateBands(elems, bp) {
  if (!bp) return;
  var total = 0;
  for (var i = 0; i < BAND_KEYS.length; i++) total += (bp[BAND_KEYS[i]] || 0);
  if (total < 1e-9) total = 1e-9;
  var domKey = 'd', domVal = 0;
  for (var bi = 0; bi < BAND_KEYS.length; bi++) {
    var k   = BAND_KEYS[bi];
    var val = bp[k] || 0;
    var pct = val / total * 100;
    elems[k].val.textContent = pct.toFixed(1) + '%';
    elems[k].bar.style.width = Math.min(pct, 100) + '%';
    elems[k].card.classList.remove('dominant');
    if (val > domVal) { domVal = val; domKey = k; }
  }
  elems[domKey].card.classList.add('dominant');
}

function onBatch(msg) {
  var q1 = msg.quality  || 'flat';
  var q2 = msg.quality2 || 'flat';

  updateQuality('qualBar1','qlIcon1','qlText1', q1, 'CH1', {
    qlRaw1:  (msg.raw_rms   || 0).toFixed(1),
    qlFilt1: (msg.filt_rms  || 0).toFixed(1),
    qlHum1:  (msg.hum_pct   || 0).toFixed(0),
    qlAB1:   (msg.ab_ratio  || 0).toFixed(2),
    qlTA1:   (msg.ta_ratio  || 0).toFixed(2)
  });
  updateQuality('qualBar2','qlIcon2','qlText2', q2, 'CH2', {
    qlFilt2: (msg.filt_rms2  || 0).toFixed(1),
    qlAB2:   (msg.ab_ratio2  || 0).toFixed(2),
    qlTA2:   (msg.ta_ratio2  || 0).toFixed(2)
  });

  if (recState === 'recording') {
    allBatches.push(msg);
    totalEpochs++;
    document.getElementById('smplTxt').textContent = totalEpochs + ' epochs';
    document.getElementById('finfo').textContent =
      totalEpochs + ' epochs  |  ' + (totalEpochs).toFixed(0) + ' s recorded';
  }

  // CH1 waveform
  var s1 = msg.ch  || [];
  for (var i = 0; i < s1.length; i++) {
    if (i % DECIM === 0) dispQ1.push(s1[i] / 10.0);
  }
  // CH2 waveform
  var s2 = msg.ch2 || [];
  for (var i = 0; i < s2.length; i++) {
    if (i % DECIM === 0) dispQ2.push(s2[i] / 10.0);
  }

  updateBands(be1, msg.bp);
  updateBands(be2, msg.bp2);
}

function dlJSON() {
  if (!allBatches.length) { alert('No data yet.'); return; }
  var doc = {
    _version: '2.0',
    device: 'BioAmp EXG Pill x2 + ESP32',
    channels: {
      ch1: { electrodes: 'Fp1(IN+) Fp2(IN-) M1(REF)', location: 'Frontal',  gpio: 34 },
      ch2: { electrodes: 'T7(IN+)  T8(IN-)  M2(REF)', location: 'Temporal', gpio: 35 }
    },
    sample_rate_hz: 256,
    epoch_samples: 256,
    value_encoding: 'int16 x10 — divide by 10.0 for uV',
    session_start_iso: new Date(sessStart).toISOString(),
    total_epochs: allBatches.length,
    batches: allBatches.map(function(b) {
      return {
        elapsed_ms: b.ts,
        epoch:      b.epoch,
        quality:    b.quality,
        quality2:   b.quality2,
        ch1:        b.ch,
        ch2:        b.ch2,
        bp1:        b.bp,
        bp2:        b.bp2,
        rel1:       b.rel,
        rel2:       b.rel2,
        ab_ratio:   b.ab_ratio,
        ta_ratio:   b.ta_ratio,
        ab_ratio2:  b.ab_ratio2,
        ta_ratio2:  b.ta_ratio2
      };
    })
  };
  var ts = new Date(sessStart).toISOString().replace(/[:.]/g,'-').slice(0,19);
  var a  = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));
  a.download = 'eeg_dual_' + ts + '.json';
  a.click();
}

function clearSession() {
  if (!confirm('Clear all recorded data?')) return;
  allBatches = []; totalEpochs = 0;
  dispQ1.length = 0; dispQ2.length = 0;
  sessStart = Date.now();
  var empty = []; for (var i = 0; i < WIN; i++) empty.push(null);
  chart1.data.datasets[0].data = empty.slice();
  chart2.data.datasets[0].data = empty.slice();
  chart1.update('none'); chart2.update('none');
  document.getElementById('smplTxt').textContent = '0 epochs';
  setRecState('idle');
}

connect();
</script>
</body>
</html>
)EEG_PAGE";
