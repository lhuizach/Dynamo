"""Dynamo training dashboard. Run: python dashboard.py"""
from __future__ import annotations
import glob
import json
import os
import subprocess
import sys
import time
from typing import Optional

from flask import Flask, jsonify, request

app = Flask(__name__)
_proc: Optional[subprocess.Popen] = None
_start_time: float = 0.0

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamo Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
header{padding:14px 24px;border-bottom:1px solid #30363d;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:18px;font-weight:700;color:#58a6ff}
.badge{font-size:12px;padding:3px 10px;border-radius:12px;font-weight:600}
.badge.running{background:#1f6a1f;color:#3fb950;border:1px solid #3fb95055}
.badge.stopped{background:#21262d;color:#8b949e;border:1px solid #30363d}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;padding:20px 24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}
.card h3{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.card .val{font-size:28px;font-weight:700;color:#e6edf3;font-variant-numeric:tabular-nums}
.card .sub{font-size:12px;color:#484f58;margin-top:4px}
.section{padding:0 24px 20px}
.section h2{font-size:14px;font-weight:600;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}
.chart-wrap{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;height:260px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 24px 20px}
.config-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}
.config-card h2{font-size:13px;font-weight:600;color:#8b949e;margin-bottom:16px;text-transform:uppercase;letter-spacing:.5px}
.field{margin-bottom:12px}
.field label{display:block;font-size:12px;color:#8b949e;margin-bottom:4px}
.field input,.field select{width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:7px 10px;border-radius:6px;font-size:13px;outline:none}
.field input:focus{border-color:#58a6ff}
.field-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn-row{display:flex;gap:8px;margin-top:16px}
button{padding:9px 20px;border-radius:6px;border:none;cursor:pointer;font-size:14px;font-weight:600;transition:opacity .15s}
button:hover:not(:disabled){opacity:.85}
button:disabled{opacity:.4;cursor:not-allowed}
.btn-start{background:#238636;color:#fff}
.btn-stop{background:#b62324;color:#fff}
.btn-resume{background:#1f6feb;color:#fff}
.ckpt-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;overflow-y:auto;max-height:320px}
.ckpt-card h2{font-size:13px;font-weight:600;color:#8b949e;margin-bottom:16px;text-transform:uppercase;letter-spacing:.5px}
.ckpt-item{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #21262d;font-size:13px}
.ckpt-item:last-child{border-bottom:none}
.ckpt-name{color:#79c0ff;font-family:monospace}
.ckpt-size{color:#484f58;font-size:11px}
.progress-wrap{padding:0 24px 20px}
.progress-bar-bg{background:#21262d;border-radius:8px;height:8px;overflow:hidden}
.progress-bar{background:linear-gradient(90deg,#1f6feb,#58a6ff);height:100%;border-radius:8px;transition:width .5s}
.progress-label{display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin-top:6px}
</style>
</head>
<body>
<header>
  <h1>⚡ Dynamo Dashboard</h1>
  <span class="badge stopped" id="statusBadge">Stopped</span>
</header>

<div class="grid">
  <div class="card"><h3>Step</h3><div class="val" id="statStep">—</div><div class="sub" id="statMaxSteps"></div></div>
  <div class="card"><h3>Loss</h3><div class="val" id="statLoss">—</div></div>
  <div class="card"><h3>Learning Rate</h3><div class="val" id="statLr">—</div></div>
  <div class="card"><h3>Throughput</h3><div class="val" id="statTok">—</div><div class="sub">tok/s</div></div>
</div>

<div class="progress-wrap">
  <div class="progress-bar-bg"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
  <div class="progress-label"><span id="progressPct">0%</span><span id="eta">—</span></div>
</div>

<div class="section">
  <h2>Loss Curve</h2>
  <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
</div>

<div class="controls">
  <div class="config-card">
    <h2>Training Config</h2>
    <div class="field-row">
      <div class="field"><label>Dim</label><input id="cDim" value="256"></div>
      <div class="field"><label>Layers</label><input id="cLayers" value="4"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Heads</label><input id="cHeads" value="4"></div>
      <div class="field"><label>KV Heads</label><input id="cKvHeads" value="2"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>FFN Dim</label><input id="cFfn" value="512"></div>
      <div class="field"><label>Seq Len</label><input id="cSeqLen" value="512"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Batch Size</label><input id="cBatch" value="2"></div>
      <div class="field"><label>Grad Accum</label><input id="cGradAccum" value="4"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Max Steps</label><input id="cMaxSteps" value="100000"></div>
      <div class="field"><label>Warmup Steps</label><input id="cWarmup" value="500"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Max LR</label><input id="cMaxLr" value="0.0003"></div>
      <div class="field"><label>Log Every</label><input id="cLogEvery" value="100"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Save Every</label><input id="cSaveEvery" value="1000"></div>
      <div class="field"><label>No BNB</label>
        <select id="cNoBnb"><option value="true" selected>Yes (AdamW)</option><option value="false">No (Adam8bit)</option></select>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn-start" id="btnStart" onclick="startTraining()">▶ Start</button>
      <button class="btn-resume" id="btnResume" onclick="resumeTraining()">↺ Resume</button>
      <button class="btn-stop" id="btnStop" onclick="stopTraining()" disabled>■ Stop</button>
    </div>
  </div>

  <div class="ckpt-card">
    <h2>Checkpoints</h2>
    <div id="ckptList"><span style="color:#484f58;font-size:13px">No checkpoints yet</span></div>
  </div>
</div>

<script>
const ctx = document.getElementById('lossChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: { labels: [], datasets: [{
    label: 'Loss', data: [],
    borderColor: '#58a6ff', backgroundColor: '#58a6ff22',
    borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3
  }]},
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#484f58', maxTicksLimit: 10 }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#484f58' }, grid: { color: '#21262d' } }
    }
  }
});

let lastStep = -1;

async function poll() {
  try {
    const [status, metrics, ckpts] = await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/metrics').then(r=>r.json()),
      fetch('/checkpoints').then(r=>r.json()),
    ]);

    // status badge
    const badge = document.getElementById('statusBadge');
    badge.textContent = status.running ? 'Running' : 'Stopped';
    badge.className = 'badge ' + (status.running ? 'running' : 'stopped');
    document.getElementById('btnStart').disabled = status.running;
    document.getElementById('btnResume').disabled = status.running;
    document.getElementById('btnStop').disabled = !status.running;

    // stats
    if (metrics.length > 0) {
      const last = metrics[metrics.length - 1];
      document.getElementById('statStep').textContent = last.step.toLocaleString();
      document.getElementById('statLoss').textContent = last.loss.toFixed(4);
      document.getElementById('statLr').textContent = last.lr.toExponential(1);
      document.getElementById('statTok').textContent = last.tok_per_sec.toLocaleString();
      document.getElementById('statMaxSteps').textContent = '/ ' + last.max_steps.toLocaleString();

      const pct = (last.step / last.max_steps * 100).toFixed(1);
      document.getElementById('progressBar').style.width = pct + '%';
      document.getElementById('progressPct').textContent = pct + '%';

      if (status.running && status.elapsed > 0 && last.step > 0) {
        const stepsPerSec = last.step / status.elapsed;
        const remaining = (last.max_steps - last.step) / stepsPerSec;
        document.getElementById('eta').textContent = 'ETA ' + fmtTime(remaining);
      }
    }

    // chart — only update on new data
    if (metrics.length > 0 && metrics[metrics.length-1].step !== lastStep) {
      lastStep = metrics[metrics.length-1].step;
      const sample = metrics.filter((_,i) => i % Math.max(1, Math.floor(metrics.length/200)) === 0);
      chart.data.labels = sample.map(m => m.step);
      chart.data.datasets[0].data = sample.map(m => m.loss);
      chart.update();
    }

    // checkpoints
    const cl = document.getElementById('ckptList');
    if (ckpts.length === 0) {
      cl.innerHTML = '<span style="color:#484f58;font-size:13px">No checkpoints yet</span>';
    } else {
      cl.innerHTML = ckpts.map(c =>
        `<div class="ckpt-item">
          <span class="ckpt-name">${c.name}</span>
          <span class="ckpt-size">${c.size}</span>
        </div>`
      ).join('');
    }
  } catch(e) { console.error(e); }
  setTimeout(poll, 2000);
}

function buildArgs(resume) {
  const noBnb = document.getElementById('cNoBnb').value === 'true';
  return {
    dim: +document.getElementById('cDim').value,
    n_layers: +document.getElementById('cLayers').value,
    n_heads: +document.getElementById('cHeads').value,
    n_kv_heads: +document.getElementById('cKvHeads').value,
    ffn_dim: +document.getElementById('cFfn').value,
    seq_len: +document.getElementById('cSeqLen').value,
    batch_size: +document.getElementById('cBatch').value,
    grad_accum: +document.getElementById('cGradAccum').value,
    max_steps: +document.getElementById('cMaxSteps').value,
    warmup_steps: +document.getElementById('cWarmup').value,
    max_lr: +document.getElementById('cMaxLr').value,
    log_every: +document.getElementById('cLogEvery').value,
    save_every: +document.getElementById('cSaveEvery').value,
    no_bnb: noBnb,
    resume: resume,
  };
}

async function startTraining() {
  await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(buildArgs(false))});
}
async function resumeTraining() {
  await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(buildArgs(true))});
}
async function stopTraining() {
  await fetch('/stop', {method:'POST'});
}

function fmtTime(sec) {
  if (sec < 60) return Math.round(sec) + 's';
  if (sec < 3600) return Math.round(sec/60) + 'm';
  const h = Math.floor(sec/3600), m = Math.round((sec%3600)/60);
  return h + 'h ' + m + 'm';
}

poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/status")
def status():
    global _proc, _start_time
    running = _proc is not None and _proc.poll() is None
    if not running:
        _proc = None
    return jsonify({"running": running, "elapsed": time.time() - _start_time if running else 0})


@app.route("/start", methods=["POST"])
def start():
    global _proc, _start_time
    if _proc is not None and _proc.poll() is None:
        return jsonify({"error": "already running"}), 400

    data = request.get_json()
    cmd = [sys.executable, "training/train.py",
        "--tokenizer", "tokenizer/dynamo.json",
        "--output", "dynamo/",
        "--dim", str(data["dim"]),
        "--n-layers", str(data["n_layers"]),
        "--n-heads", str(data["n_heads"]),
        "--n-kv-heads", str(data["n_kv_heads"]),
        "--ffn-dim", str(data["ffn_dim"]),
        "--seq-len", str(data["seq_len"]),
        "--batch-size", str(data["batch_size"]),
        "--grad-accum", str(data["grad_accum"]),
        "--max-steps", str(data["max_steps"]),
        "--warmup-steps", str(data["warmup_steps"]),
        "--max-lr", str(data["max_lr"]),
        "--log-every", str(data["log_every"]),
        "--save-every", str(data["save_every"]),
    ]
    if data.get("no_bnb"):
        cmd.append("--no-bnb")
    if data.get("resume"):
        cmd.append("--resume")

    _proc = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    _start_time = time.time()
    return jsonify({"started": True})


@app.route("/stop", methods=["POST"])
def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
    return jsonify({"stopped": True})


@app.route("/metrics")
def metrics():
    log_path = os.path.join("dynamo", "training_log.jsonl")
    if not os.path.exists(log_path):
        return jsonify([])
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return jsonify(rows)


@app.route("/checkpoints")
def checkpoints():
    results = []
    for path in sorted(glob.glob("dynamo/checkpoint*.pt"), reverse=True):
        size_mb = os.path.getsize(path) / 1e6
        results.append({"name": os.path.basename(path), "size": f"{size_mb:.0f} MB"})
    return jsonify(results)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5001)
    args = p.parse_args()
    print(f"Dashboard → http://localhost:{args.port}")
    app.run(port=args.port, debug=False)
