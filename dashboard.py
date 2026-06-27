"""Dynamo training dashboard. Run: python dashboard.py"""
from __future__ import annotations
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, request

app = Flask(__name__)
_proc: Optional[subprocess.Popen] = None
_start_time: float = 0.0
_last_config: dict = {}
_schedules: list[dict] = []
_schedule_id: int = 0
_scheduler_running: bool = True

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

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
.section h2{font-size:13px;font-weight:600;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}
.chart-wrap{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;height:260px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 24px 20px}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:0 24px 20px}
.panel{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}
.panel h2{font-size:13px;font-weight:600;color:#8b949e;margin-bottom:16px;text-transform:uppercase;letter-spacing:.5px}
.field{margin-bottom:12px}
.field label{display:block;font-size:12px;color:#8b949e;margin-bottom:4px}
.field input,.field select{width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:7px 10px;border-radius:6px;font-size:13px;outline:none}
.field input:focus,.field select:focus{border-color:#58a6ff}
.field-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn-row{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap}
button{padding:8px 18px;border-radius:6px;border:none;cursor:pointer;font-size:13px;font-weight:600;transition:opacity .15s}
button:hover:not(:disabled){opacity:.85}
button:disabled{opacity:.4;cursor:not-allowed}
.btn-start{background:#238636;color:#fff}
.btn-stop{background:#b62324;color:#fff}
.btn-resume{background:#1f6feb;color:#fff}
.btn-sm{padding:5px 12px;font-size:12px;background:#21262d;color:#c9d1d9;border:1px solid #30363d}
.btn-danger{background:#6e1a1a;color:#f85149;border:1px solid #6e1a1a}
.progress-wrap{padding:0 24px 16px}
.progress-bar-bg{background:#21262d;border-radius:8px;height:8px;overflow:hidden}
.progress-bar{background:linear-gradient(90deg,#1f6feb,#58a6ff);height:100%;border-radius:8px;transition:width .5s}
.progress-label{display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin-top:6px}
.ckpt-item{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #21262d;font-size:13px}
.ckpt-item:last-child{border-bottom:none}
.ckpt-name{color:#79c0ff;font-family:monospace;font-size:12px}
.ckpt-size{color:#484f58;font-size:11px}
.sched-item{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #21262d;font-size:13px}
.sched-item:last-child{border-bottom:none}
.sched-action{font-weight:600;margin-right:8px}
.sched-action.start{color:#3fb950}
.sched-action.stop{color:#f85149}
.sched-time{color:#e6edf3;font-family:monospace}
.sched-days{color:#8b949e;font-size:11px;margin-left:8px}
.days-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.day-btn{padding:4px 8px;border-radius:4px;border:1px solid #30363d;background:#0d1117;color:#8b949e;cursor:pointer;font-size:11px;font-weight:600;transition:all .15s}
.day-btn.active{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff}
.empty{color:#484f58;font-size:13px;font-style:italic}
.toggle-row{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.toggle{position:relative;width:36px;height:20px}
.toggle input{opacity:0;width:0;height:0}
.slider{position:absolute;cursor:pointer;inset:0;background:#30363d;border-radius:20px;transition:.3s}
.slider:before{content:"";position:absolute;height:14px;width:14px;left:3px;bottom:3px;background:#8b949e;border-radius:50%;transition:.3s}
input:checked+.slider{background:#238636}
input:checked+.slider:before{transform:translateX(16px);background:#fff}
.toggle-label{font-size:13px;color:#8b949e}
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

<div class="two-col">
  <!-- Training Config -->
  <div class="panel">
    <h2>Training Config</h2>
    <div class="field-row">
      <div class="field"><label>Dim</label><input id="cDim" value="2048"></div>
      <div class="field"><label>Layers</label><input id="cLayers" value="16"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Heads</label><input id="cHeads" value="16"></div>
      <div class="field"><label>KV Heads</label><input id="cKvHeads" value="4"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>FFN Dim</label><input id="cFfn" value="5632"></div>
      <div class="field"><label>Seq Len</label><input id="cSeqLen" value="4096"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Batch Size</label><input id="cBatch" value="1"></div>
      <div class="field"><label>Grad Accum</label><input id="cGradAccum" value="16"></div>
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
      <div class="field"><label>Optimizer</label>
        <select id="cNoBnb"><option value="false">Adam8bit (bnb)</option><option value="true">AdamW (fallback)</option></select>
      </div>
    </div>
    <div class="field-row">
      <div class="field"><label>torch.compile</label>
        <select id="cCompile"><option value="false" selected>Off</option><option value="true">On (+20-40% speed)</option></select>
      </div>
      <div class="field"><label>TF32</label><input value="enabled (auto)" disabled style="color:#484f58"></div>
    </div>
    <div class="btn-row">
      <button class="btn-start" id="btnStart" onclick="startTraining()">▶ Start</button>
      <button class="btn-resume" id="btnResume" onclick="resumeTraining()">↺ Resume</button>
      <button class="btn-stop" id="btnStop" onclick="stopTraining()" disabled>■ Stop</button>
    </div>
  </div>

  <!-- Checkpoints -->
  <div class="panel" style="overflow-y:auto;max-height:480px">
    <h2>Checkpoints</h2>
    <div id="ckptList"><span class="empty">No checkpoints yet</span></div>
  </div>
</div>

<!-- Scheduler -->
<div class="section">
  <h2>Scheduler</h2>
</div>
<div class="two-col" style="margin-top:-12px">
  <div class="panel">
    <div class="toggle-row">
      <label class="toggle"><input type="checkbox" id="schedEnabled" onchange="toggleScheduler()" checked><span class="slider"></span></label>
      <span class="toggle-label" id="schedLabel">Scheduler enabled</span>
    </div>
    <h2 style="margin-bottom:12px">Add Schedule</h2>
    <div class="field-row">
      <div class="field"><label>Action</label>
        <select id="sAction">
          <option value="start">▶ Start training</option>
          <option value="stop">■ Stop training</option>
        </select>
      </div>
      <div class="field"><label>Time (24h)</label><input type="time" id="sTime" value="22:00"></div>
    </div>
    <div class="field">
      <label>Days (none = every day)</label>
      <div class="days-row" id="dayBtns">
        <span class="day-btn" onclick="toggleDay(this,0)">Mon</span>
        <span class="day-btn" onclick="toggleDay(this,1)">Tue</span>
        <span class="day-btn" onclick="toggleDay(this,2)">Wed</span>
        <span class="day-btn" onclick="toggleDay(this,3)">Thu</span>
        <span class="day-btn" onclick="toggleDay(this,4)">Fri</span>
        <span class="day-btn" onclick="toggleDay(this,5)">Sat</span>
        <span class="day-btn" onclick="toggleDay(this,6)">Sun</span>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn-start" onclick="addSchedule()">+ Add</button>
    </div>
  </div>

  <div class="panel" style="overflow-y:auto;max-height:280px">
    <h2>Active Schedules</h2>
    <div id="schedList"><span class="empty">No schedules yet</span></div>
  </div>
</div>

<script>
const ctx = document.getElementById('lossChart').getContext('2d');
const chart = new Chart(ctx, {
  type:'line',
  data:{labels:[],datasets:[{label:'Loss',data:[],borderColor:'#58a6ff',backgroundColor:'#58a6ff22',borderWidth:2,pointRadius:0,fill:true,tension:0.3}]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#484f58',maxTicksLimit:10},grid:{color:'#21262d'}},y:{ticks:{color:'#484f58'},grid:{color:'#21262d'}}}}
});

let lastStep = -1;
let selectedDays = [];

function toggleDay(el, day) {
  el.classList.toggle('active');
  if (selectedDays.includes(day)) selectedDays = selectedDays.filter(d=>d!==day);
  else selectedDays.push(day);
}

async function poll() {
  try {
    const [status, metrics, ckpts, scheds] = await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/metrics').then(r=>r.json()),
      fetch('/checkpoints').then(r=>r.json()),
      fetch('/schedules').then(r=>r.json()),
    ]);

    const badge = document.getElementById('statusBadge');
    badge.textContent = status.running ? 'Running' : 'Stopped';
    badge.className = 'badge '+(status.running?'running':'stopped');
    document.getElementById('btnStart').disabled = status.running;
    document.getElementById('btnResume').disabled = status.running;
    document.getElementById('btnStop').disabled = !status.running;

    if (metrics.length > 0) {
      const last = metrics[metrics.length-1];
      document.getElementById('statStep').textContent = last.step.toLocaleString();
      document.getElementById('statLoss').textContent = last.loss.toFixed(4);
      document.getElementById('statLr').textContent = last.lr.toExponential(1);
      document.getElementById('statTok').textContent = last.tok_per_sec.toLocaleString();
      document.getElementById('statMaxSteps').textContent = '/ '+last.max_steps.toLocaleString();
      const pct = (last.step/last.max_steps*100).toFixed(1);
      document.getElementById('progressBar').style.width = pct+'%';
      document.getElementById('progressPct').textContent = pct+'%';
      if (status.running && status.elapsed > 0 && last.step > 0) {
        const eta = (last.max_steps - last.step) / (last.step / status.elapsed);
        document.getElementById('eta').textContent = 'ETA '+fmtTime(eta);
      }
    }

    if (metrics.length > 0 && metrics[metrics.length-1].step !== lastStep) {
      lastStep = metrics[metrics.length-1].step;
      const sample = metrics.filter((_,i)=>i%Math.max(1,Math.floor(metrics.length/200))===0);
      chart.data.labels = sample.map(m=>m.step);
      chart.data.datasets[0].data = sample.map(m=>m.loss);
      chart.update();
    }

    const cl = document.getElementById('ckptList');
    cl.innerHTML = ckpts.length === 0
      ? '<span class="empty">No checkpoints yet</span>'
      : ckpts.map(c=>`<div class="ckpt-item"><span class="ckpt-name">${c.name}</span><span class="ckpt-size">${c.size}</span></div>`).join('');

    const sl = document.getElementById('schedList');
    sl.innerHTML = scheds.length === 0
      ? '<span class="empty">No schedules yet</span>'
      : scheds.map(s=>`
        <div class="sched-item">
          <div>
            <span class="sched-action ${s.action}">${s.action==='start'?'▶ Start':'■ Stop'}</span>
            <span class="sched-time">${s.time}</span>
            <span class="sched-days">${s.days.length?s.days.map(d=>['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d]).join(', '):'Every day'}</span>
          </div>
          <button class="btn-sm btn-danger" onclick="deleteSchedule(${s.id})">✕</button>
        </div>`).join('');
  } catch(e) { console.error(e); }
  setTimeout(poll, 2000);
}

function buildArgs(resume) {
  return {
    dim:+document.getElementById('cDim').value,
    n_layers:+document.getElementById('cLayers').value,
    n_heads:+document.getElementById('cHeads').value,
    n_kv_heads:+document.getElementById('cKvHeads').value,
    ffn_dim:+document.getElementById('cFfn').value,
    seq_len:+document.getElementById('cSeqLen').value,
    batch_size:+document.getElementById('cBatch').value,
    grad_accum:+document.getElementById('cGradAccum').value,
    max_steps:+document.getElementById('cMaxSteps').value,
    warmup_steps:+document.getElementById('cWarmup').value,
    max_lr:+document.getElementById('cMaxLr').value,
    log_every:+document.getElementById('cLogEvery').value,
    save_every:+document.getElementById('cSaveEvery').value,
    no_bnb: document.getElementById('cNoBnb').value==='true',
    compile: document.getElementById('cCompile').value==='true',
    resume,
  };
}

async function startTraining()  { await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(buildArgs(false))}); }
async function resumeTraining() { await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(buildArgs(true))}); }
async function stopTraining()   { await fetch('/stop',{method:'POST'}); }

async function addSchedule() {
  const time = document.getElementById('sTime').value;
  const action = document.getElementById('sAction').value;
  if (!time) return;
  await fetch('/schedules', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action, time, days: selectedDays.slice()})});
}

async function deleteSchedule(id) {
  await fetch('/schedules/'+id, {method:'DELETE'});
}

async function toggleScheduler() {
  const enabled = document.getElementById('schedEnabled').checked;
  document.getElementById('schedLabel').textContent = enabled ? 'Scheduler enabled' : 'Scheduler disabled';
  await fetch('/schedules/toggle', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({enabled})});
}

function fmtTime(sec) {
  if (sec < 60) return Math.round(sec)+'s';
  if (sec < 3600) return Math.round(sec/60)+'m';
  const h=Math.floor(sec/3600), m=Math.round((sec%3600)/60);
  return h+'h '+m+'m';
}

poll();
</script>
</body>
</html>"""


def _scheduler_loop() -> None:
    while _scheduler_running:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()

        for sched in list(_schedules):
            if sched["time"] != current_time:
                continue
            if sched["days"] and current_day not in sched["days"]:
                continue
            if sched["action"] == "start":
                _do_start(_last_config.copy() if _last_config else {}, resume=True)
            elif sched["action"] == "stop":
                _do_stop()

        time.sleep(60 - datetime.now().second)  # sleep until next minute boundary


def _do_start(config: dict, resume: bool = False) -> None:
    global _proc, _start_time, _last_config
    if _proc is not None and _proc.poll() is None:
        return
    if not config:
        return
    _last_config = config
    cmd = [sys.executable, "training/train.py",
        "--tokenizer", "tokenizer/dynamo.json", "--output", "dynamo/",
        "--dim", str(config.get("dim", 2048)),
        "--n-layers", str(config.get("n_layers", 16)),
        "--n-heads", str(config.get("n_heads", 16)),
        "--n-kv-heads", str(config.get("n_kv_heads", 4)),
        "--ffn-dim", str(config.get("ffn_dim", 5632)),
        "--seq-len", str(config.get("seq_len", 4096)),
        "--batch-size", str(config.get("batch_size", 1)),
        "--grad-accum", str(config.get("grad_accum", 16)),
        "--max-steps", str(config.get("max_steps", 100_000)),
        "--warmup-steps", str(config.get("warmup_steps", 500)),
        "--max-lr", str(config.get("max_lr", 3e-4)),
        "--log-every", str(config.get("log_every", 100)),
        "--save-every", str(config.get("save_every", 1000)),
    ]
    if config.get("no_bnb", False):
        cmd.append("--no-bnb")
    if config.get("compile", False):
        cmd.append("--compile")
    if resume:
        cmd.append("--resume")
    _proc = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    _start_time = time.time()


def _do_stop() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()


@app.route("/")
def index():
    return HTML


@app.route("/status")
def status():
    global _proc
    running = _proc is not None and _proc.poll() is None
    if not running:
        _proc = None
    return jsonify({"running": running, "elapsed": time.time() - _start_time if running else 0})


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json()
    resume = data.pop("resume", False)
    _do_start(data, resume=resume)
    return jsonify({"started": True})


@app.route("/stop", methods=["POST"])
def stop():
    _do_stop()
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


@app.route("/schedules", methods=["GET"])
def get_schedules():
    return jsonify(_schedules)


@app.route("/schedules", methods=["POST"])
def add_schedule():
    global _schedule_id
    data = request.get_json()
    _schedule_id += 1
    _schedules.append({
        "id": _schedule_id,
        "action": data["action"],
        "time": data["time"],
        "days": data.get("days", []),
    })
    return jsonify({"id": _schedule_id})


@app.route("/schedules/<int:sid>", methods=["DELETE"])
def delete_schedule(sid: int):
    global _schedules
    _schedules = [s for s in _schedules if s["id"] != sid]
    return jsonify({"deleted": True})


@app.route("/schedules/toggle", methods=["POST"])
def toggle_scheduler():
    global _scheduler_running
    _scheduler_running = request.get_json().get("enabled", True)
    return jsonify({"enabled": _scheduler_running})


if __name__ == "__main__":
    import argparse
    import socket
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5001)
    args = p.parse_args()

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()

    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"Dashboard → http://localhost:{args.port}")
    print(f"On your phone → http://{local_ip}:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
