"""Dynamo web UI — single file. Run: python server.py"""
from __future__ import annotations
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class _HideGets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /" not in record.getMessage()

logging.getLogger("werkzeug").addFilter(_HideGets())

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamo</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#21262d;
  --border:#30363d;--border2:#3d444d;
  --text:#e6edf3;--muted:#8b949e;--hint:#484f58;
  --accent:#3fb950;--accent2:#58a6ff;--accent3:#bc8cff;
  --red:#f85149;--yellow:#d29922;
  --font-mono:'Cascadia Code','Fira Code','JetBrains Mono',Consolas,monospace;
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* header */
header{height:48px;border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0}
.logo{display:flex;align-items:center;gap:8px;font-weight:600;font-size:15px;color:var(--text)}
.logo-icon{width:24px;height:24px;background:var(--accent);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#0d1117}
.model-badge{font-size:11px;padding:2px 8px;background:var(--surface2);border:1px solid var(--border);border-radius:20px;color:var(--muted);font-family:var(--font-mono)}
.header-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 6px var(--accent)}
.status-dot.loading{background:var(--yellow);animation:pulse 1s infinite}
.status-dot.error{background:var(--red)}
.status-text{font-size:12px;color:var(--muted)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* layout */
.layout{flex:1;display:grid;grid-template-columns:280px 1fr;overflow:hidden}

/* sidebar */
.sidebar{border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.sidebar-section{padding:12px;border-bottom:1px solid var(--border)}
.sidebar-label{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--hint);margin-bottom:10px}
.field{margin-bottom:12px}
.field label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}
.field input[type=range]{width:100%;accent-color:var(--accent2)}
.val-row{display:flex;justify-content:space-between;align-items:center}
.val-row .val{font-size:12px;font-family:var(--font-mono);color:var(--accent2)}
input[type=number]{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-size:13px;font-family:var(--font-mono);outline:none}
input[type=number]:focus{border-color:var(--accent2)}

.examples{flex:1;overflow-y:auto;padding:12px}
.example{padding:8px 10px;border-radius:6px;font-size:12px;color:var(--muted);cursor:pointer;border:1px solid transparent;margin-bottom:4px;line-height:1.4}
.example:hover{background:var(--surface2);border-color:var(--border);color:var(--text)}
.example code{font-family:var(--font-mono);color:var(--accent3);font-size:11px}

/* main */
.main{display:flex;flex-direction:column;overflow:hidden}

/* output area */
.output-area{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.placeholder-wrap{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:var(--hint)}
.placeholder-icon{font-size:40px;opacity:.3}
.placeholder-text{font-size:14px;text-align:center;line-height:1.6}
.placeholder-text kbd{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:1px 5px;font-size:11px;font-family:var(--font-mono);color:var(--muted)}

/* message blocks */
.block{border-radius:10px;border:1px solid var(--border);overflow:hidden}
.block-header{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:var(--surface);border-bottom:1px solid var(--border)}
.block-label{font-size:11px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.block-label.router{color:var(--accent3)}
.block-label.code{color:var(--accent)}
.block-label.prompt-lbl{color:var(--accent2)}
.block-actions{display:flex;gap:6px}
.icon-btn{background:none;border:1px solid var(--border);color:var(--muted);padding:3px 8px;border-radius:5px;font-size:11px;cursor:pointer;display:flex;align-items:center;gap:4px;transition:all .15s}
.icon-btn:hover{border-color:var(--border2);color:var(--text);background:var(--surface2)}
.icon-btn.copied{border-color:var(--accent);color:var(--accent)}
.block-body{padding:0}
.block-body pre{margin:0;background:var(--surface)!important;border-radius:0!important;font-size:13px!important;line-height:1.6!important;padding:14px 16px!important;overflow-x:auto}
.router-text{padding:12px 16px;font-size:13px;color:var(--muted);line-height:1.6;font-style:italic;background:var(--surface)}
.prompt-text{padding:12px 16px;font-size:13px;color:var(--text);line-height:1.6;background:var(--surface);font-family:var(--font-mono)}
.meta-row{display:flex;gap:16px;padding:8px 12px;border-top:1px solid var(--border);background:var(--surface)}
.meta{font-size:11px;color:var(--hint);font-family:var(--font-mono)}
.meta span{color:var(--muted)}

/* loading */
.loading-block{border-radius:10px;border:1px solid var(--border);padding:16px;display:flex;align-items:center;gap:12px;background:var(--surface)}
.spinner{width:16px;height:16px;border:2px solid var(--border2);border-top-color:var(--accent2);border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
.loading-text{font-size:13px;color:var(--muted)}
@keyframes spin{to{transform:rotate(360deg)}}

/* input bar */
.input-bar{border-top:1px solid var(--border);padding:12px 16px;background:var(--surface);flex-shrink:0}
.input-wrap{display:flex;gap:8px;align-items:flex-end}
.prompt-input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:10px 14px;border-radius:8px;font-size:14px;font-family:inherit;resize:none;min-height:44px;max-height:160px;outline:none;line-height:1.5;transition:border-color .15s}
.prompt-input:focus{border-color:var(--accent2)}
.prompt-input::placeholder{color:var(--hint)}
.send-btn{height:44px;padding:0 16px;background:var(--accent2);color:#0d1117;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;flex-shrink:0;transition:opacity .15s;display:flex;align-items:center;gap:6px}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.4;cursor:not-allowed}
.send-btn svg{width:14px;height:14px}
.hint-row{margin-top:6px;display:flex;justify-content:space-between;align-items:center}
.hint-text{font-size:11px;color:var(--hint)}
.hint-text kbd{background:var(--surface2);border:1px solid var(--border);border-radius:3px;padding:0 4px;font-size:10px;font-family:var(--font-mono)}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">D</div>
    Dynamo
  </div>
  <span class="model-badge" id="modelBadge">loading…</span>
  <div class="header-right">
    <div class="status-dot" id="statusDot"></div>
    <span class="status-text" id="statusText">Ready</span>
  </div>
</header>

<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-section">
      <div class="sidebar-label">Settings</div>
      <div class="field">
        <div class="val-row">
          <label>Temperature</label>
          <span class="val" id="tempVal">0.80</span>
        </div>
        <input type="range" id="temp" min="0.1" max="2" step="0.05" value="0.8"
               oninput="tempVal.textContent=parseFloat(this.value).toFixed(2)">
      </div>
      <div class="field">
        <div class="val-row">
          <label>Repetition penalty</label>
          <span class="val" id="repVal">1.30</span>
        </div>
        <input type="range" id="rep" min="1" max="2" step="0.05" value="1.3"
               oninput="repVal.textContent=parseFloat(this.value).toFixed(2)">
      </div>
      <div class="field">
        <label>Max tokens</label>
        <input type="number" id="maxTok" value="300" min="10" max="2048">
      </div>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Quick prompts</div>
    </div>
    <div class="examples">
      <div class="example" onclick="useExample(this)">make a <code>retry</code> function with delay</div>
      <div class="example" onclick="useExample(this)">binary search in an array</div>
      <div class="example" onclick="useExample(this)">responsive navbar in HTML and CSS</div>
      <div class="example" onclick="useExample(this)">debounce function in JavaScript</div>
      <div class="example" onclick="useExample(this)">fetch JSON from API with error handling</div>
      <div class="example" onclick="useExample(this)">linked list in C</div>
      <div class="example" onclick="useExample(this)">hash map with open addressing in C</div>
      <div class="example" onclick="useExample(this)">Python class for a TTL cache</div>
      <div class="example" onclick="useExample(this)">center a div with flexbox</div>
      <div class="example" onclick="useExample(this)">dark mode toggle with CSS variables</div>
      <div class="example" onclick="useExample(this)">parse a .env config file</div>
      <div class="example" onclick="useExample(this)">event emitter class in JavaScript</div>
    </div>
  </aside>

  <main class="main">
    <div class="output-area" id="outputArea">
      <div class="placeholder-wrap" id="placeholder">
        <div class="placeholder-icon">&#9881;</div>
        <div class="placeholder-text">
          Type a coding request below and press <kbd>Ctrl+Enter</kbd><br>
          or click Generate to see Dynamo in action.
        </div>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-wrap">
        <textarea class="prompt-input" id="prompt" rows="1"
          placeholder="Ask for code… e.g. make a retry function"></textarea>
        <button class="send-btn" id="sendBtn" onclick="generate()">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 8H2M8 2l6 6-6 6"/>
          </svg>
          Generate
        </button>
      </div>
      <div class="hint-row">
        <span class="hint-text"><kbd>Ctrl+Enter</kbd> to generate</span>
        <span class="hint-text" id="genMeta"></span>
      </div>
    </div>
  </main>
</div>

<script>
fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('modelBadge').textContent = `${d.params} · ${d.device}`;
});

const prompt = document.getElementById('prompt');

prompt.addEventListener('input', () => {
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 160) + 'px';
});

prompt.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') generate();
});

function useExample(el) {
  prompt.value = el.textContent.trim();
  prompt.style.height = 'auto';
  prompt.dispatchEvent(new Event('input'));
  prompt.focus();
}

function setStatus(state, text) {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  dot.className = 'status-dot' + (state === 'loading' ? ' loading' : state === 'error' ? ' error' : '');
  txt.textContent = text;
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function copyCode(btn, code) {
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

function detectLang(code) {
  if (/<[a-z][\s\S]*>/i.test(code)) return 'html';
  if (/^\s*[.#][a-z]/m.test(code) && /{/.test(code)) return 'css';
  if (/function\s|const\s|let\s|var\s|=>/.test(code)) return 'javascript';
  if (/#include|int main|printf/.test(code)) return 'c';
  if (/def |import |print\(/.test(code)) return 'python';
  return 'plaintext';
}

async function generate() {
  const text = prompt.value.trim();
  if (!text) return;

  const area = document.getElementById('outputArea');
  document.getElementById('placeholder')?.remove();

  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  setStatus('loading', 'Generating…');

  const loader = document.createElement('div');
  loader.className = 'loading-block';
  loader.innerHTML = '<div class="spinner"></div><span class="loading-text">Thinking…</span>';
  area.appendChild(loader);
  area.scrollTop = area.scrollHeight;

  const t0 = Date.now();

  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        prompt: text,
        temperature: parseFloat(document.getElementById('temp').value),
        max_tokens: parseInt(document.getElementById('maxTok').value),
        repetition_penalty: parseFloat(document.getElementById('rep').value),
      })
    });
    const data = await resp.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    loader.remove();

    const wrap = document.createElement('div');
    wrap.style.display = 'flex';
    wrap.style.flexDirection = 'column';
    wrap.style.gap = '10px';

    // prompt block
    const promptBlock = `
      <div class="block">
        <div class="block-header">
          <span class="block-label prompt-lbl">Prompt</span>
        </div>
        <div class="block-body">
          <div class="prompt-text">${esc(data.prompt)}</div>
        </div>
      </div>`;

    // code block
    const lang = detectLang(data.generated);
    const highlighted = hljs.highlight(data.generated, {language: lang, ignoreIllegals: true}).value;
    const codeId = 'code-' + Date.now();
    const codeBlock = `
      <div class="block">
        <div class="block-header">
          <span class="block-label code">Output · ${lang}</span>
          <div class="block-actions">
            <button class="icon-btn" onclick="copyCode(this, document.getElementById('${codeId}').textContent)">Copy</button>
          </div>
        </div>
        <div class="block-body">
          <pre><code class="hljs" id="${codeId}">${highlighted}</code></pre>
        </div>
        <div class="meta-row">
          <span class="meta">${data.tokens} tokens</span>
          <span class="meta">${elapsed}s</span>
          <span class="meta">${Math.round(data.tokens / parseFloat(elapsed))} tok/s</span>
        </div>
      </div>`;

    wrap.innerHTML = promptBlock + codeBlock;
    area.appendChild(wrap);
    area.scrollTop = area.scrollHeight;

    document.getElementById('genMeta').textContent = `${data.tokens} tokens · ${elapsed}s`;
    setStatus('ready', 'Ready');
  } catch(e) {
    loader.remove();
    const err = document.createElement('div');
    err.className = 'block';
    err.innerHTML = `<div class="block-header"><span class="block-label" style="color:var(--red)">Error</span></div><div class="router-text" style="color:var(--red)">${esc(String(e))}</div>`;
    area.appendChild(err);
    setStatus('error', 'Error');
  }

  btn.disabled = false;
}
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
        max_tokens: int = int(data.get("max_tokens", 300))
        rep_penalty: float = float(data.get("repetition_penalty", 1.3))

        ids = tok.encode(prompt)
        idx = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_id=tok.eos_id,
                repetition_penalty=rep_penalty,
            )
        gen_ids = out[0][len(ids):].tolist()
        return jsonify({"prompt": prompt, "generated": tok.decode(gen_ids), "tokens": len(gen_ids)})

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
