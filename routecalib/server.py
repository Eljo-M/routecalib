"""Zero-dependency OpenAI-compatible HTTP routing proxy + interactive live dashboard."""

from __future__ import annotations

import json
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from routecalib.router import CalibratedRouter
from routecalib.schemas import RouteRequest, dumps_strict_json

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>RouteCalib — Calibrated Multi-Model Router</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 24px; color: #f8fafc; }
  .sub { color: #94a3b8; margin-bottom: 20px; font-size: 14px; }
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
  label { display: block; font-size: 12px; color: #cbd5e1; margin: 10px 0 4px; }
  textarea, input, select { width: 100%; box-sizing: border-box; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 8px; font-size: 13px; }
  textarea { min-height: 110px; font-family: ui-monospace, monospace; }
  button { margin-top: 12px; background: #0284c7; color: #fff; border: none; border-radius: 6px; padding: 10px 16px; font-weight: 600; cursor: pointer; }
  button:hover { background: #0369a1; }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }
  .kpi { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; }
  .kpi .v { font-size: 18px; font-weight: 700; color: #38bdf8; }
  .kpi .k { font-size: 11px; color: #94a3b8; }
  pre { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; overflow: auto; font-size: 12px; max-height: 290px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>RouteCalib Live Proxy</h1>
  <div class="sub">Conformal-calibrated multi-model routing with SLA enforcement, semantic caching, and cost telemetry.</div>
  <div class="grid">
    <div class="card">
      <label for="prompt">Prompt</label>
      <textarea id="prompt">Extract the company name and date as JSON from: Acme Corp reported Q3 results on 2026-10-12.</textarea>
      <label for="alpha">Target False-Accept Risk Budget (alpha: <span id="alphaVal">0.05</span>)</label>
      <input id="alpha" type="range" min="0.02" max="0.25" step="0.01" value="0.05" oninput="document.getElementById('alphaVal').textContent=this.value" />
      <label for="sla">Max Latency SLA in ms (optional)</label>
      <input id="sla" type="number" placeholder="e.g. 600 (leave blank for none)" />
      <label><input id="cache" type="checkbox" checked style="width:auto;margin-right:6px;" />Enable bounded semantic cache</label>
      <button onclick="sendPrompt()">Route &amp; Execute</button>
      <label>Latest Routing Decision</label>
      <pre id="decision">Submit a prompt to inspect conformal scores and routing decision.</pre>
    </div>
    <div class="card">
      <div class="kpis">
        <div class="kpi"><div class="v" id="kReq">0</div><div class="k">Requests</div></div>
        <div class="kpi"><div class="v" id="kSav">0%</div><div class="k">Cost Saved</div></div>
        <div class="kpi"><div class="v" id="kOv">0 ms</div><div class="k">p95 Overhead</div></div>
      </div>
      <label>Live Telemetry (/metrics)</label>
      <pre id="metrics">Loading...</pre>
    </div>
  </div>
</div>
<script>
async function refreshMetrics() {
  const r = await fetch('/metrics');
  const m = await r.json();
  document.getElementById('kReq').textContent = m.total_requests;
  document.getElementById('kSav').textContent = m.estimated_cost_savings_pct + '%';
  document.getElementById('kOv').textContent = m.router_overhead_p95_ms + ' ms';
  document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);
}
async function sendPrompt() {
  const slaRaw = document.getElementById('sla').value.trim();
  const payload = {
    messages: [{ role: 'user', content: document.getElementById('prompt').value }],
    target_alpha: parseFloat(document.getElementById('alpha').value),
    allow_cache: document.getElementById('cache').checked,
    max_latency_ms: slaRaw ? parseFloat(slaRaw) : null
  };
  const r = await fetch('/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await r.json();
  document.getElementById('decision').textContent = JSON.stringify(data.x_routecalib || data, null, 2);
  await refreshMetrics();
}
refreshMetrics();
</script>
</body>
</html>
"""


def create_server(router: CalibratedRouter, *, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer bound to the supplied CalibratedRouter."""

    class RouteCalibHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # Keep server output clean in tests & CLI

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
