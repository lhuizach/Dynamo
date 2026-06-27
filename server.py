"""Dynamo web UI — single file. Run: python server.py"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:14px 24px;border-bottom:1px solid #30363d;display:flex;align-items:center;gap:10px}
header h1{font-size:18px;font-weight:700;color:#58a6ff;letter-spacing:-.3px}
.badge{font-size:11px;padding:2px 8px;background:#1f6feb22;border:1px solid #1f6feb55;border-radius:12px;color:#79c0ff}
.controls{padding:10px 24px;border-bottom:1px solid #30363d;display:flex;gap:28px;align-items:center;font-size:13px;color:#8b949e}
.controls label{display:flex;align-items:center;gap:8px}
.controls input[type=range]{width:90px;accent-color:#58a6ff}
.controls input[type=number]{width:60px;background:#161b22;border:1px solid #30363d;color:#e6edf3;padding:3px 7px;border-radius:6px;font-size:13px}
.val{color:#58a6ff;min-width:30px}
main{flex:1;display:flex;flex-direction:column;padding:20px 24px 0;gap:12px;overflow:hidden}
textarea{width:100%;background:#161b22;border:1px solid #30363d;border-radius:8px;color:#e6edf3;font-family:'Cascadia Code','Fira Code',Consolas,monospace;font-size:14px;line-height:1.6;padding:12px 14px;resize:vertical;min-height:80px;max-height:180px;outline:none;transition:border-color .15s}
textarea:focus{border-color:#58a6ff}
.btn-row{display:flex;gap:8px;margin-top:8px}
button{padding:7px 18px;border-radius:6px;border:none;cursor:pointer;font-size:14px;font-weight:500;transition:opacity .15s}
button:hover{opacity:.85}
button:disabled{opacity:.4;cursor:not-allowed}
.btn-gen{background:#238636;color:#fff}
.btn-clear{background:#21262d;color:#c9d1d9;border:1px solid #30363d}
.out-wrap{flex:1;overflow-y:auto;background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:0;min-height:0}
.out{padding:16px;font-family:'Cascadia Code','Fira Code',Consolas,monospace;font-size:13px;line-height:1.75;white-space:pre-wrap;word-break:break-word}
.out .p{color:#8b949e}
.out .g{color:#79c0ff}
.out .ph{color:#484f58;font-style:italic;font-family:sans-serif;font-size:13px}
.spinner{display:inline-block;width:13px;height:13px;border:2px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.status{font-size:12px;color:#484f58;padding:6px 24px 10px;min-height:24px}
</style>
</head>
<body>
<header>
  <h1>⚡ Dynamo</h1>
  <span class="badge" id="badge">loading…</span>
</header>
<div class="controls">
  <label>Temperature
    <input type="range" id="temp" min="0.1" max="2" step="0.05" value="0.8"
           oninput="tempVal.textContent=parseFloat(this.value).toFixed(2)">
    <span class="val" id="tempVal">0.80</span>
  </label>
  <label>Max tokens
    <input type="number" id="maxTok" value="200" min="10" max="1024">
  </label>
</div>
<main>
  <div>
    <textarea id="prompt" placeholder="Enter a code prompt…   e.g.  def retry(&#10;&#10;Ctrl+Enter to generate" rows="4"></textarea>
    <div class="btn-row">
      <button class="btn-gen" id="genBtn" onclick="generate()">Generate</button>
      <button class="btn-clear" onclick="clearAll()">Clear</button>
    </div>
  </div>
  <div class="out-wrap">
    <div class="out" id="out"><span class="ph">Output will appear here…</span></div>
  </div>
</main>
<div class="status" id="status"></div>

<script>
fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('badge').textContent = d.params+'  '+d.device;
});

document.getElementById('prompt').addEventListener('keydown', e=>{
  if(e.ctrlKey && e.key==='Enter') generate();
});

async function generate(){
  const prompt = document.getElementById('prompt').value;
  if(!prompt.trim()) return;
  const temp  = parseFloat(document.getElementById('temp').value);
  const maxTok = parseInt(document.getElementById('maxTok').value);
  const btn = document.getElementById('genBtn');
  const out = document.getElementById('out');
  const status = document.getElementById('status');

  btn.disabled = true;
  out.innerHTML = '<span class="spinner"></span><span style="color:#8b949e">Generating…</span>';
  status.textContent = '';

  try {
    const t0 = Date.now();
    const resp = await fetch('/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt, temperature:temp, max_tokens:maxTok})
    });
    const d = await resp.json();
    const sec = ((Date.now()-t0)/1000).toFixed(1);
    out.innerHTML =
      '<span class="p">'+esc(d.prompt)+'</span>'+
      '<span class="g">'+esc(d.generated)+'</span>';
    status.textContent = d.tokens+' tokens · '+sec+'s';
  } catch(e) {
    out.innerHTML = '<span style="color:#f85149">'+esc(String(e))+'</span>';
  }
  btn.disabled = false;
}

function clearAll(){
  document.getElementById('prompt').value='';
  document.getElementById('out').innerHTML='<span class="ph">Output will appear here…</span>';
  document.getElementById('status').textContent='';
}

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }
</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamo web UI")
    p.add_argument("--checkpoint", default="dynamo/checkpoint_final.pt")
    p.add_argument("--tokenizer", default="tokenizer/dynamo.json")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    import torch
    from flask import Flask, jsonify, request
    from architecture.model import Dynamo
    from architecture.tokenizer import DynamoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = DynamoTokenizer(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Dynamo(ckpt["config"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_count = sum(p.numel() for p in set(model.parameters()))
    print(f"Dynamo loaded — {param_count/1e6:.1f}M params on {dev}")
    print(f"Open → http://localhost:{args.port}")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return HTML

    @app.route("/info")
    def info():
        return jsonify({"params": f"{param_count/1e6:.1f}M", "device": dev})

    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.get_json()
        prompt: str = data.get("prompt", "")
        temperature: float = float(data.get("temperature", 0.8))
        max_tokens: int = int(data.get("max_tokens", 200))

        ids = tok.encode(prompt)
        idx = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_id=tok.eos_id,
            )
        gen_ids = out[0][len(ids):].tolist()
        return jsonify({"prompt": prompt, "generated": tok.decode(gen_ids), "tokens": len(gen_ids)})

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
