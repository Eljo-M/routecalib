"""Zero-dependency OpenAI-compatible HTTP routing proxy + high-tier interactive control plane UI."""

from __future__ import annotations

import json
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from routecalib.router import CalibratedRouter
from routecalib.schemas import RouteRequest, dumps_strict_json

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>RouteCalib — Calibrated Multi-Model Router Control Plane</title>
<style>
  :root {
    --bg: #070b14;
    --panel: rgba(15, 23, 42, 0.82);
    --panel-solid: #0f172a;
    --border: rgba(148, 163, 184, 0.16);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    background:
      radial-gradient(900px 500px at 12% -10%, rgba(56, 189, 248, 0.14), transparent 60%),
      radial-gradient(800px 500px at 90% 0%, rgba(129, 140, 248, 0.13), transparent 60%),
      var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 24px 20px 48px;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  .topbar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding-bottom: 18px; margin-bottom: 20px; border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #0284c7, #6366f1);
    display: grid; place-items: center; font-weight: 800; font-size: 16px; color: #fff;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.35);
  }
  .brand h1 { margin: 0; font-size: 20px; letter-spacing: -0.02em; }
  .brand p { margin: 2px 0 0; font-size: 12.5px; color: var(--muted); }
  .status-pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
    border-radius: 999px; background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.3); color: var(--green); font-size: 12px; font-weight: 600;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); }
  .tabs { display: flex; gap: 8px; margin-bottom: 18px; }
  .tab {
    padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.6); color: var(--muted); font-size: 13px; font-weight: 600; cursor: pointer;
  }
  .tab.active { background: rgba(56, 189, 248, 0.14); border-color: rgba(56, 189, 248, 0.45); color: #e0f2fe; }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
  @media (max-width: 820px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
  .kpi {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; backdrop-filter: blur(10px);
  }
  .kpi .k { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi .v { font-size: 24px; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em; }
  .kpi .s { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .grid { display: grid; grid-template-columns: 1.25fr 1fr; gap: 18px; align-items: start; }
  @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px; backdrop-filter: blur(12px);
  }
  .card h2 { margin: 0 0 12px; font-size: 14px; font-weight: 700; letter-spacing: 0.01em; color: #e2e8f0; }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  .preset-btn {
    background: rgba(30, 41, 59, 0.85); color: #cbd5e1; border: 1px solid var(--border);
    border-radius: 999px; padding: 5px 11px; font-size: 11.5px; font-weight: 500; cursor: pointer;
    transition: all 0.15s ease;
  }
  .preset-btn:hover { border-color: var(--accent); color: #fff; background: rgba(56, 189, 248, 0.14); }
  label { display: block; font-size: 12px; color: #cbd5e1; margin: 12px 0 6px; font-weight: 500; }
  textarea, input[type="number"], select {
    width: 100%; background: #090e1a; color: var(--text); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 12px; font-size: 13px; outline: none;
  }
  textarea { min-height: 112px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.45; resize: vertical; }
  textarea:focus, input[type="number"]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15); }
  .controls-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 6px; }
  .range-wrap { display: flex; align-items: center; gap: 10px; }
  input[type="range"] { flex: 1; accent-color: var(--accent); }
  .alpha-badge {
    min-width: 48px; text-align: center; padding: 3px 8px; border-radius: 6px;
    background: rgba(56, 189, 248, 0.15); color: var(--accent); font-weight: 700; font-size: 12px;
  }
  .actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
  .check-lbl { display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 12.5px; color: #cbd5e1; cursor: pointer; }
  .primary-btn {
    background: linear-gradient(135deg, #0284c7, #4f46e5); color: #fff; border: none;
    border-radius: 10px; padding: 10px 18px; font-size: 13px; font-weight: 700; cursor: pointer;
    box-shadow: 0 8px 20px rgba(2, 132, 199, 0.3);
  }
  .primary-btn:hover { filter: brightness(1.08); }
  .decision-banner {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    padding: 12px 14px; border-radius: 10px; background: #090e1a; border: 1px solid var(--border); margin-bottom: 14px;
  }
  .tier-pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 700;
  }
  .pill-small { background: rgba(52, 211, 153, 0.16); color: var(--green); border: 1px solid rgba(52, 211, 153, 0.35); }
  .pill-mid { background: rgba(251, 191, 36, 0.16); color: var(--amber); border: 1px solid rgba(251, 191, 36, 0.35); }
  .pill-frontier { background: rgba(129, 140, 248, 0.18); color: #c7d2fe; border: 1px solid rgba(129, 140, 248, 0.4); }
  .pill-cache { background: rgba(56, 189, 248, 0.18); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.45); }
  .prob-bars { display: flex; flex-direction: column; gap: 10px; margin: 10px 0 14px; }
  .bar-row .bar-head { display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1; margin-bottom: 4px; }
  .bar-track { height: 10px; border-radius: 999px; background: #090e1a; border: 1px solid var(--border); overflow: hidden; position: relative; }
  .bar-fill { height: 100%; border-radius: 999px; transition: width 0.25s ease; }
  .history { display: flex; flex-direction: column; gap: 8px; max-height: 280px; overflow-y: auto; }
  .hist-item {
    padding: 10px 12px; border-radius: 10px; background: #090e1a; border: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 12px;
  }
  .hist-prompt { color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 240px; }
  pre {
    background: #090e1a; border: 1px solid var(--border); border-radius: 10px;
    padding: 12px; overflow: auto; font-size: 11.5px; color: #cbd5e1; max-height: 220px; margin: 0;
  }
  .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
  .plot-card img { width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #fff; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="logo">RC</div>
      <div>
        <h1>RouteCalib Control Plane</h1>
        <p>Conformal-Calibrated Multi-Model Router · SLA Guard · Numeric-Safe Semantic Cache</p>
      </div>
    </div>
    <div class="status-pill"><span class="dot"></span>Live Proxy · OpenAI-Compatible API</div>
  </div>

  <div class="tabs">
    <button class="tab active" id="tabLive" onclick="switchTab('live')">Live Routing Playground</button>
    <button class="tab" id="tabBench" onclick="switchTab('bench')">Offline Benchmarks &amp; Proof</button>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="k">Total Routed</div><div class="v" id="kReq">0</div><div class="s">Live proxy requests</div></div>
    <div class="kpi"><div class="k">Cost Saved vs Frontier</div><div class="v" style="color:var(--green)" id="kSav">0.0%</div><div class="s" id="kSpend">$0.0000 spent</div></div>
    <div class="kpi"><div class="k">Cache Hit Rate</div><div class="v" style="color:var(--accent)" id="kCache">0.0%</div><div class="s" id="kCacheHits">0 hits (numeric-guarded)</div></div>
    <div class="kpi"><div class="k">Router p95 Overhead</div><div class="v" style="color:var(--accent-2)" id="kOv">0.00 ms</div><div class="s">CPU feature + conformal gate</div></div>
  </div>

  <div id="viewLive" class="grid">
    <div class="card">
      <h2>1. Test Realistic Workloads</h2>
      <div class="presets">
        <button class="preset-btn" onclick="setPreset('extract')">Easy JSON Extraction</button>
        <button class="preset-btn" onclick="setPreset('qa')">Short CS Q&amp;A</button>
        <button class="preset-btn" onclick="setPreset('reason')">Systems Trade-off</button>
        <button class="preset-btn" onclick="setPreset('math')">Conformal Proof (Math)</button>
        <button class="preset-btn" onclick="setPreset('code')">Async Deadlock (Coding)</button>
        <button class="preset-btn" onclick="setPreset('inv104')">Invoice #104 (Cache)</button>
        <button class="preset-btn" onclick="setPreset('inv105')">Invoice #105 (No Collision)</button>
      </div>
      <label for="prompt">User Prompt</label>
      <textarea id="prompt">Extract the company name and date as JSON from: Acme Corp reported Q3 results on 2026-10-12.</textarea>

      <div class="controls-row">
        <div>
          <label for="alpha">Target False-Accept Risk Budget (&alpha;)</label>
          <div class="range-wrap">
            <input id="alpha" type="range" min="0.02" max="0.25" step="0.01" value="0.05" oninput="document.getElementById('alphaVal').textContent=parseFloat(this.value).toFixed(2)" />
            <span class="alpha-badge" id="alphaVal">0.05</span>
          </div>
        </div>
        <div>
          <label for="sla">Hard Latency SLA (ms, optional)</label>
          <input id="sla" type="number" placeholder="e.g. 500 (blank = unconstrained)" />
        </div>
      </div>

      <div class="actions">
        <label class="check-lbl"><input id="cache" type="checkbox" checked />Enable bounded semantic cache (numeric-guarded)</label>
        <button class="primary-btn" onclick="sendPrompt()">Route &amp; Execute &rarr;</button>
      </div>

      <h2 style="margin-top:20px;">2. Model Response</h2>
      <pre id="responseBox">Click "Route & Execute" to run a calibrated routing decision.</pre>
    </div>

    <div style="display:flex;flex-direction:column;gap:18px;">
      <div class="card">
        <h2>Conformal Routing Decision</h2>
        <div class="decision-banner">
          <div>
            <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">SELECTED TIER</div>
            <span id="tierBadge" class="tier-pill pill-frontier">Waiting for request</span>
          </div>
          <div style="text-align:right;">
            <div style="font-size:11px;color:var(--muted);">DOMAIN · OVERHEAD</div>
            <div id="domMeta" style="font-size:13px;font-weight:600;margin-top:2px;">—</div>
          </div>
        </div>

        <div class="prob-bars" id="probBars">
          <div class="bar-row">
            <div class="bar-head"><span>Small Local (3B) P(pass)</span><span id="pSmall">—</span></div>
            <div class="bar-track"><div class="bar-fill" id="bSmall" style="width:0%;background:var(--green)"></div></div>
          </div>
          <div class="bar-row">
            <div class="bar-head"><span>Mid-Size (8B–14B) P(pass)</span><span id="pMid">—</span></div>
            <div class="bar-track"><div class="bar-fill" id="bMid" style="width:0%;background:var(--amber)"></div></div>
          </div>
        </div>

        <label style="margin-top:6px;">Escalation / Conformal Trace</label>
        <pre id="decisionTrace">{}</pre>
      </div>

      <div class="card">
        <h2>Recent Request Feed</h2>
        <div class="history" id="histFeed">
          <div style="font-size:12px;color:var(--muted);">No requests routed yet.</div>
        </div>
      </div>
    </div>
  </div>

  <div id="viewBench" class="hidden">
    <div class="plots-grid">
      <div class="card plot-card">
        <h2>Cost–Accuracy Pareto Frontier</h2>
        <img src="/results/pareto_cost_accuracy.png" alt="Pareto Frontier" />
      </div>
      <div class="card plot-card">
        <h2>Conformal Risk Control (Global vs Mondrian)</h2>
        <img src="/results/conformal_coverage.png" alt="Conformal Coverage" />
      </div>
      <div class="card plot-card">
        <h2>Shadow-Traffic CUSUM Drift Detection</h2>
        <img src="/results/drift_detection.png" alt="Drift Detection" />
      </div>
    </div>
  </div>
</div>
<script>
const PRESETS = {
  extract: "Extract the company name and date as JSON from: Acme Corp reported Q3 results on 2026-10-12.",
  qa: "Summarize the difference between TCP and UDP in two sentences.",
  reason: "Compare and contrast Raft and Multi-Paxos leader election under asymmetric network partitions; analyze edge cases where stale leaders serve linearizable reads.",
  math: "Prove that for any n >= 1, the finite-sample split conformal quantile ceil((n+1)(1-alpha))/n achieves marginal coverage >= 1 - alpha under exchangeability; derive the exact upper bound.",
  code: "Debug this async Python connection pool where a cancelled coroutine leaks a semaphore permit and causes intermittent deadlock under timeout; refactor with strict cancellation invariants and pytest tests.\\n```python\\nasync def acquire(self):\\n    await self._sem.acquire()\\n```",
  inv104: "Extract vendor, total, and status as JSON for invoice 104 from Acme Corp.",
  inv105: "Extract vendor, total, and status as JSON for invoice 105 from Acme Corp."
};
const historyItems = [];

function setPreset(key) {
  document.getElementById('prompt').value = PRESETS[key];
}

function switchTab(name) {
  document.getElementById('viewLive').classList.toggle('hidden', name !== 'live');
  document.getElementById('viewBench').classList.toggle('hidden', name !== 'bench');
  document.getElementById('tabLive').classList.toggle('active', name === 'live');
  document.getElementById('tabBench').classList.toggle('active', name === 'bench');
}

function pillClass(tier, cacheHit) {
  if (cacheHit) return 'tier-pill pill-cache';
  if (tier === 'tier0_small') return 'tier-pill pill-small';
  if (tier === 'tier1_mid') return 'tier-pill pill-mid';
  return 'tier-pill pill-frontier';
}

async function refreshMetrics() {
  const r = await fetch('/metrics');
  const m = await r.json();
  document.getElementById('kReq').textContent = m.total_requests;
  document.getElementById('kSav').textContent = m.estimated_cost_savings_pct.toFixed(1) + '%';
  document.getElementById('kSpend').textContent = '$' + m.total_cost_usd.toFixed(5) + ' spent vs $' + m.frontier_equivalent_cost_usd.toFixed(5);
  const hitRate = m.cache ? (m.cache.hit_rate * 100).toFixed(1) : '0.0';
  document.getElementById('kCache').textContent = hitRate + '%';
  document.getElementById('kCacheHits').textContent = (m.cache_hits || 0) + ' hits (numeric-guarded)';
  document.getElementById('kOv').textContent = m.router_overhead_p95_ms.toFixed(2) + ' ms';
}

async function sendPrompt() {
  const promptText = document.getElementById('prompt').value;
  const slaRaw = document.getElementById('sla').value.trim();
  const payload = {
    messages: [{ role: 'user', content: promptText }],
    target_alpha: parseFloat(document.getElementById('alpha').value),
    allow_cache: document.getElementById('cache').checked,
    max_latency_ms: slaRaw ? parseFloat(slaRaw) : null
  };
  const r = await fetch('/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await r.json();
  const dec = data.x_routecalib || {};
  const msg = data.choices && data.choices[0] ? data.choices[0].message.content : JSON.stringify(data);
  document.getElementById('responseBox').textContent = msg;

  const badge = document.getElementById('tierBadge');
  badge.className = pillClass(dec.selected_tier, dec.cache_hit);
  badge.textContent = dec.cache_hit ? 'CACHE HIT (' + dec.selected_tier + ')' : (dec.selected_tier || 'error');
  document.getElementById('domMeta').textContent = (dec.inferred_domain || '—') + ' · ' + (dec.router_overhead_ms || 0).toFixed(3) + ' ms';

  const probs = dec.tier_success_probs || {};
  const ps = probs.tier0_small !== undefined ? (probs.tier0_small * 100) : (dec.cache_hit ? 100 : 0);
  const pm = probs.tier1_mid !== undefined ? (probs.tier1_mid * 100) : (dec.cache_hit ? 100 : 0);
  document.getElementById('pSmall').textContent = probs.tier0_small !== undefined ? ps.toFixed(1) + '%' : (dec.cache_hit ? 'cached' : '—');
  document.getElementById('pMid').textContent = probs.tier1_mid !== undefined ? pm.toFixed(1) + '%' : (dec.cache_hit ? 'cached' : '—');
  document.getElementById('bSmall').style.width = ps.toFixed(1) + '%';
  document.getElementById('bMid').style.width = pm.toFixed(1) + '%';
  document.getElementById('decisionTrace').textContent = JSON.stringify(dec, null, 2);

  historyItems.unshift({
    prompt: promptText,
    tier: dec.cache_hit ? 'CACHE HIT' : dec.selected_tier,
    cacheHit: !!dec.cache_hit,
    rawTier: dec.selected_tier,
    overhead: (dec.router_overhead_ms || 0).toFixed(2) + ' ms'
  });
  const feed = document.getElementById('histFeed');
  feed.innerHTML = historyItems.slice(0, 10).map(item =>
    '<div class="hist-item">' +
      '<span class="hist-prompt">' + item.prompt.replace(/</g, '&lt;') + '</span>' +
      '<div style="display:flex;align-items:center;gap:8px;">' +
        '<span style="color:var(--muted)">' + item.overhead + '</span>' +
        '<span class="' + pillClass(item.rawTier, item.cacheHit) + '">' + item.tier + '</span>' +
      '</div>' +
    '</div>'
  ).join('');
  await refreshMetrics();
}
refreshMetrics();
</script>
</body>
</html>
"""


def create_server(
    router: CalibratedRouter,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    results_dir: Path | None = None,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer bound to the supplied CalibratedRouter."""
    static_results = Path("results") if results_dir is None else Path(results_dir)

    class RouteCalibHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = dumps_strict_json(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path == "/index.html":
                raw = DASHBOARD_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "tiers": [t.tier_id for t in router.tiers]})
            elif self.path == "/metrics":
                self._send_json(HTTPStatus.OK, router.metrics_snapshot())
            elif self.path.startswith("/results/") and self.path.endswith(".png"):
                filename = Path(self.path).name
                target = (static_results / filename).resolve()
                if target.parent == static_results.resolve() and target.is_file():
                    raw = target.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Plot not found: {filename}"})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = body.get("messages", [])
                prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("content")).strip()
                if not prompt:
                    prompt = str(body.get("prompt", "")).strip()
                req = RouteRequest(
                    request_id=str(body.get("request_id") or f"req-{uuid.uuid4().hex[:10]}"),
                    prompt=prompt,
                    domain=str(body.get("domain", "auto")),
                    max_latency_ms=body.get("max_latency_ms"),
                    max_cost_usd=body.get("max_cost_usd"),
                    target_alpha=float(body.get("target_alpha", 0.05)),
                    allow_cache=bool(body.get("allow_cache", True)),
                    expected_output_tokens=int(body.get("max_tokens", 160)),
                )
                decision, resp = router.route_and_execute(req)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "id": req.request_id,
                        "object": "chat.completion",
                        "model": decision.selected_tier,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": resp.text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": resp.input_tokens,
                            "completion_tokens": resp.output_tokens,
                            "total_tokens": resp.input_tokens + resp.output_tokens,
                        },
                        "x_routecalib": decision.to_dict(),
                    },
                )
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), RouteCalibHandler)
