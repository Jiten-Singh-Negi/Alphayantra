"""
dashboard/app.py
─────────────────
Serves the professional web dashboard at http://localhost:8000/dashboard

Features:
  - Live NIFTY/BANKNIFTY price ticker
  - Real-time options chain (PCR, Max Pain, OI heatmap)
  - Top ML signals with confidence bars
  - Risk manager status panel
  - Intraday equity curve
  - Telegram alert toggle
  - Dark/light mode
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaYantra — Professional Trading Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --purple: #bc8cff; --orange: #ffa657;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:13px; }

  /* ── Top bar ── */
  .topbar { display:flex; align-items:center; justify-content:space-between;
            padding:10px 20px; background:var(--surface); border-bottom:1px solid var(--border);
            position:sticky; top:0; z-index:100; }
  .logo { font-size:18px; font-weight:700; color:var(--accent); letter-spacing:1px; }
  .market-status { display:flex; gap:16px; align-items:center; }
  .status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }
  .dot-green { background:var(--green); box-shadow:0 0 6px var(--green); animation:pulse 2s infinite; }
  .dot-red   { background:var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Index ticker strip ── */
  .ticker-strip { display:flex; gap:0; overflow:hidden; background:#0a0f16;
                  border-bottom:1px solid var(--border); }
  .ticker-item { padding:6px 24px; border-right:1px solid var(--border);
                 display:flex; gap:12px; align-items:center; white-space:nowrap; }
  .ticker-name { font-weight:600; color:var(--muted); font-size:11px; letter-spacing:.5px; }
  .ticker-price { font-weight:700; font-size:15px; font-variant-numeric:tabular-nums; }
  .up { color:var(--green); } .down { color:var(--red); }
  .ticker-change { font-size:11px; }

  /* ── Main layout ── */
  .main { display:grid; grid-template-columns:1fr 360px; gap:0; height:calc(100vh - 88px); overflow:hidden; }
  .left-panel  { overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:16px; }
  .right-panel { overflow-y:auto; border-left:1px solid var(--border); padding:16px;
                 display:flex; flex-direction:column; gap:16px; background:var(--surface); }

  /* ── Cards ── */
  .card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; }
  .card-title { font-size:11px; font-weight:600; color:var(--muted); letter-spacing:.8px;
                text-transform:uppercase; margin-bottom:12px; display:flex;
                align-items:center; justify-content:space-between; }
  .card-title .badge { font-size:10px; padding:2px 8px; border-radius:10px;
                       background:#1c2128; border:1px solid var(--border); }

  /* ── Signal cards ── */
  .signal-list { display:flex; flex-direction:column; gap:8px; }
  .signal-item { display:grid; grid-template-columns:auto 1fr auto auto;
                 gap:8px; align-items:center; padding:10px 12px;
                 background:#0d1117; border-radius:6px; border:1px solid var(--border); }
  .signal-item:hover { border-color:var(--accent); }
  .signal-badge { font-size:10px; font-weight:700; padding:3px 8px; border-radius:4px; white-space:nowrap; }
  .badge-strong-buy  { background:#0d2a1a; color:var(--green); border:1px solid var(--green); }
  .badge-buy         { background:#0d1f12; color:#7ee787; border:1px solid #3fb95066; }
  .badge-hold        { background:#1c1f24; color:var(--muted); border:1px solid var(--border); }
  .badge-sell        { background:#2a0d0d; color:#ff7b72; border:1px solid #f8514966; }
  .badge-strong-sell { background:#3a0a0a; color:var(--red); border:1px solid var(--red); }
  .signal-ticker { font-weight:700; font-size:14px; }
  .signal-price  { font-size:13px; font-variant-numeric:tabular-nums; }
  .confidence-bar { height:4px; background:#21262d; border-radius:2px; margin-top:3px; width:100%; }
  .confidence-fill { height:100%; border-radius:2px; transition:width .5s; }
  .signal-meta { font-size:10px; color:var(--muted); }

  /* ── Options chain table ── */
  .options-grid { display:grid; grid-template-columns:1fr 80px 1fr; gap:0; font-size:11px; }
  .options-header { display:contents; }
  .options-header > div { padding:6px 4px; background:#0a0f16; font-weight:600;
                          color:var(--muted); text-align:center; border-bottom:1px solid var(--border); }
  .options-row { display:contents; }
  .options-row > div { padding:5px 4px; border-bottom:1px solid #1c2128;
                       text-align:center; font-variant-numeric:tabular-nums; }
  .options-row:hover > div { background:#1c2128; }
  .atm-row > div { background:#161b22; font-weight:700; color:var(--yellow) !important; }
  .ce-side { color:#7ee787; }
  .pe-side { color:#ff7b72; }
  .strike-col { color:var(--text); font-weight:600; background:#0d1117 !important; }
  .oi-bar { height:3px; background:#21262d; border-radius:1px; margin-top:2px; }
  .oi-bar-fill { height:100%; border-radius:1px; }

  /* ── Metrics grid ── */
  .metrics-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .metric-item { background:#0d1117; border-radius:6px; padding:10px 12px;
                 border:1px solid var(--border); }
  .metric-label { font-size:10px; color:var(--muted); margin-bottom:4px; }
  .metric-value { font-size:18px; font-weight:700; font-variant-numeric:tabular-nums; }
  .metric-sub   { font-size:10px; color:var(--muted); margin-top:2px; }

  /* ── Risk panel ── */
  .risk-bar { height:8px; background:#21262d; border-radius:4px; margin:4px 0 8px; overflow:hidden; }
  .risk-bar-fill { height:100%; border-radius:4px; transition:width 1s; }
  .risk-row { display:flex; justify-content:space-between; align-items:center;
              padding:6px 0; border-bottom:1px solid #1c2128; }
  .risk-row:last-child { border-bottom:none; }
  .kill-switch { padding:8px 16px; border-radius:6px; border:1px solid;
                 cursor:pointer; font-size:12px; font-weight:600; width:100%;
                 margin-top:8px; }
  .kill-on  { background:#2a0d0d; color:var(--red);  border-color:var(--red); }
  .kill-off { background:#0d2a1a; color:var(--green); border-color:var(--green); }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

  /* ── Live dot ── */
  .live-indicator { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--muted); }
</style>
</head>
<body>

<!-- Top Bar -->
<div class="topbar">
  <div class="logo">⚡ AlphaYantra</div>
  <div class="market-status">
    <div class="live-indicator">
      <span class="status-dot dot-green" id="marketDot"></span>
      <span id="marketStatus">Loading...</span>
    </div>
    <span style="color:var(--muted)" id="currentTime"></span>
  </div>
</div>

<!-- Index Ticker Strip -->
<div class="ticker-strip">
  <div class="ticker-item">
    <span class="ticker-name">NIFTY 50</span>
    <span class="ticker-price up" id="nifty-price">--</span>
    <span class="ticker-change up" id="nifty-change">--</span>
  </div>
  <div class="ticker-item">
    <span class="ticker-name">BANKNIFTY</span>
    <span class="ticker-price" id="banknifty-price">--</span>
    <span class="ticker-change" id="banknifty-change">--</span>
  </div>
  <div class="ticker-item">
    <span class="ticker-name">FINNIFTY</span>
    <span class="ticker-price" id="finnifty-price">--</span>
    <span class="ticker-change" id="finnifty-change">--</span>
  </div>
  <div class="ticker-item">
    <span class="ticker-name">INDIA VIX</span>
    <span class="ticker-price" id="vix-price">--</span>
    <span class="ticker-change" id="vix-change">--</span>
  </div>
</div>

<!-- Main Layout -->
<div class="main">
  <!-- Left: Signals + Options -->
  <div class="left-panel">

    <!-- Top ML Signals -->
    <div class="card">
      <div class="card-title">
        🎯 Top ML Signals
        <span class="badge" id="signal-count">Loading...</span>
      </div>
      <div class="signal-list" id="signal-list">
        <div style="color:var(--muted);text-align:center;padding:20px">
          Loading signals... ensure API server is running on port 8000
        </div>
      </div>
    </div>

    <!-- Options Chain -->
    <div class="card">
      <div class="card-title">
        📊 NIFTY Options Chain
        <div style="display:flex;gap:8px;align-items:center">
          <select id="optSymbol" style="background:#0d1117;border:1px solid var(--border);
            color:var(--text);padding:2px 6px;border-radius:4px;font-size:11px">
            <option value="NIFTY">NIFTY</option>
            <option value="BANKNIFTY">BANKNIFTY</option>
          </select>
          <span class="badge" id="pcr-badge">PCR --</span>
        </div>
      </div>

      <!-- OI Summary -->
      <div class="metrics-grid" style="margin-bottom:12px">
        <div class="metric-item">
          <div class="metric-label">Put-Call Ratio</div>
          <div class="metric-value" id="opt-pcr">--</div>
          <div class="metric-sub" id="opt-sentiment">--</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Max Pain</div>
          <div class="metric-value" id="opt-maxpain">--</div>
          <div class="metric-sub" id="opt-spot">Spot: --</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">ATM IV Rank</div>
          <div class="metric-value" id="opt-ivrank">--</div>
          <div class="metric-sub">0=Low IV, 100=High IV</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">OI Ratio CE:PE</div>
          <div class="metric-value" id="opt-oiratio">--</div>
          <div class="metric-sub" id="opt-support">Support: --</div>
        </div>
      </div>

      <!-- Chain Table -->
      <div style="overflow-x:auto">
        <div class="options-grid" id="options-chain">
          <div>CE OI | LTP | IV</div>
          <div>Strike</div>
          <div>IV | LTP | PE OI</div>
          <div style="color:var(--muted);text-align:center;grid-column:1/-1;padding:20px">
            Loading options chain...
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- Right Panel: Risk + Metrics -->
  <div class="right-panel">

    <!-- Risk Status -->
    <div class="card">
      <div class="card-title">🛡️ Risk Manager</div>
      <div class="risk-row">
        <span style="color:var(--muted)">Daily P&L</span>
        <span id="risk-pnl" style="font-weight:700">₹0</span>
      </div>
      <div class="risk-bar">
        <div class="risk-bar-fill" id="risk-pnl-bar" style="width:0%;background:var(--green)"></div>
      </div>
      <div class="risk-row">
        <span style="color:var(--muted)">Capital Deployed</span>
        <span id="risk-deployed">0%</span>
      </div>
      <div class="risk-bar">
        <div class="risk-bar-fill" id="risk-deploy-bar" style="width:0%;background:var(--accent)"></div>
      </div>
      <div class="risk-row">
        <span style="color:var(--muted)">Drawdown</span>
        <span id="risk-drawdown">0%</span>
      </div>
      <div class="risk-row">
        <span style="color:var(--muted)">Trades Today</span>
        <span id="risk-trades">0</span>
      </div>
      <div class="risk-row">
        <span style="color:var(--muted)">Open Positions</span>
        <span id="risk-positions">0</span>
      </div>
      <div class="risk-row">
        <span style="color:var(--muted)">Status</span>
        <span id="risk-status" style="color:var(--green)">Active</span>
      </div>
      <button class="kill-switch kill-off" id="kill-btn" onclick="toggleKillSwitch()">
        🟢 Trading Active — Click to Emergency Stop
      </button>
    </div>

    <!-- Market Metrics -->
    <div class="card">
      <div class="card-title">📈 Market Snapshot</div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-label">Advance / Decline</div>
          <div class="metric-value" id="adv-dec">-- / --</div>
          <div class="metric-sub">NSE stocks</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">India VIX</div>
          <div class="metric-value" id="vix-val">--</div>
          <div class="metric-sub" id="vix-note">--</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Strong Buys</div>
          <div class="metric-value up" id="strong-buys">--</div>
          <div class="metric-sub">Today's signals</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Strong Sells</div>
          <div class="metric-value down" id="strong-sells">--</div>
          <div class="metric-sub">Today's signals</div>
        </div>
      </div>
    </div>

    <!-- OI Support/Resistance -->
    <div class="card">
      <div class="card-title">🧱 OI Support / Resistance</div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Based on open interest clustering</div>
      <div id="sr-levels">
        <div style="color:var(--muted);text-align:center;padding:12px">Loading...</div>
      </div>
    </div>

    <!-- Model Status -->
    <div class="card">
      <div class="card-title">🧠 ML Model Status</div>
      <div id="model-status">
        <div class="risk-row">
          <span style="color:var(--muted)">Status</span>
          <span style="color:var(--green)">Trained</span>
        </div>
        <div class="risk-row">
          <span style="color:var(--muted)">Ensemble AUC</span>
          <span id="model-auc">--</span>
        </div>
        <div class="risk-row">
          <span style="color:var(--muted)">LSTM</span>
          <span id="model-lstm">--</span>
        </div>
        <div class="risk-row">
          <span style="color:var(--muted)">Last Trained</span>
          <span id="model-date" style="font-size:11px">--</span>
        </div>
        <div class="risk-row">
          <span style="color:var(--muted)">Next Retrain</span>
          <span style="font-size:11px;color:var(--muted)">Sunday 11:00 PM</span>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
// ── Config ─────────────────────────────────────────────────────────────
// Dynamic host — works on localhost, LAN, or any cloud server
const _host  = window.location.hostname || 'localhost';
const _proto = window.location.protocol === 'https:' ? 'https' : 'http';
const _ws    = window.location.protocol === 'https:' ? 'wss'   : 'ws';
const API    = `${_proto}://${_host}:8000`;
const WS_URL = `${_ws}://${_host}:8765`;

// ── Clock ──────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', {timeZone:'Asia/Kolkata'}));
  document.getElementById('currentTime').textContent =
    ist.toLocaleTimeString('en-IN', {hour:'2-digit',minute:'2-digit',second:'2-digit'}) + ' IST';
  const h = ist.getHours(), m = ist.getMinutes();
  const open = (h > 9 || (h === 9 && m >= 15)) && (h < 15 || (h === 15 && m <= 30));
  const dot  = document.getElementById('marketDot');
  document.getElementById('marketStatus').textContent = open ? 'Market Open' : 'Market Closed';
  dot.className = 'status-dot ' + (open ? 'dot-green' : 'dot-red');
}
setInterval(updateClock, 1000); updateClock();

// ── Helpers ────────────────────────────────────────────────────────────
const fmt  = n => '₹' + (+n).toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
const fmtN = n => (+n).toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});

function signalBadge(sig) {
  const map = {
    'STRONG BUY':  ['badge-strong-buy',  '🚀 STRONG BUY'],
    'BUY':         ['badge-buy',          '🟢 BUY'],
    'HOLD':        ['badge-hold',         '⚪ HOLD'],
    'SELL':        ['badge-sell',         '🔴 SELL'],
    'STRONG SELL': ['badge-strong-sell',  '💥 STRONG SELL'],
  };
  const [cls, label] = map[sig] || ['badge-hold', sig];
  return `<span class="signal-badge ${cls}">${label}</span>`;
}

// ── Load Signals ───────────────────────────────────────────────────────
async function loadSignals() {
  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({strategy:{name:'Dashboard'},universe:'nifty50',top_n:15})
    });
    if (!res.ok) return;
    const data = await res.json();
    renderSignals(data.predictions || []);
  } catch(e) {
    document.getElementById('signal-list').innerHTML =
      '<div style="color:var(--red);text-align:center;padding:20px">API server not reachable — run: python run.py</div>';
  }
}

function renderSignals(signals) {
  const el = document.getElementById('signal-list');
  document.getElementById('signal-count').textContent = signals.length + ' signals';

  let strongBuys = 0, strongSells = 0;
  el.innerHTML = signals.map(s => {
    if (s.signal === 'STRONG BUY')  strongBuys++;
    if (s.signal === 'STRONG SELL') strongSells++;
    const conf = s.ml_confidence || 50;
    const barColor = conf > 70 ? '#3fb950' : conf > 55 ? '#58a6ff' : '#d29922';
    const chg = ((s.close - (s.close / (1 + (s.change_pct||0)/100))) ).toFixed(0);
    return `
    <div class="signal-item">
      ${signalBadge(s.signal)}
      <div>
        <div class="signal-ticker">${s.ticker}</div>
        <div class="signal-meta">Tech: ${s.tech_score?.toFixed(0)||'--'} | News: ${s.news_score?.toFixed(0)||'--'} | ${s.confirmations||0} confirms</div>
      </div>
      <div style="text-align:right">
        <div class="signal-price">${fmt(s.close)}</div>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width:${conf}%;background:${barColor}"></div>
        </div>
        <div class="signal-meta">${conf.toFixed(0)}% ML</div>
      </div>
      <div style="text-align:right;font-size:11px">
        <div style="color:var(--red)">SL: ${fmt(s.stop_loss)}</div>
        <div style="color:var(--green)">TP: ${fmt(s.take_profit)}</div>
      </div>
    </div>`;
  }).join('');

  document.getElementById('strong-buys').textContent  = strongBuys;
  document.getElementById('strong-sells').textContent = strongSells;
}

// ── Load Options Chain ─────────────────────────────────────────────────
async function loadOptions() {
  const symbol = document.getElementById('optSymbol').value;
  try {
    const res  = await fetch(`${API}/options/chain?symbol=${symbol}`);
    if (!res.ok) return;
    const data = await res.json();
    renderOptions(data);
  } catch(e) { console.log('Options load error:', e); }
}

function renderOptions(data) {
  const pcr   = data.pcr || 0;
  const color = pcr > 1.2 ? 'var(--green)' : pcr < 0.8 ? 'var(--red)' : 'var(--yellow)';

  document.getElementById('opt-pcr').textContent       = pcr.toFixed(2);
  document.getElementById('opt-pcr').style.color       = color;
  document.getElementById('opt-sentiment').textContent = data.sentiment || '--';
  document.getElementById('opt-maxpain').textContent   = data.max_pain ? fmtN(data.max_pain) : '--';
  document.getElementById('opt-spot').textContent      = 'Spot: ' + (data.spot_price ? fmtN(data.spot_price) : '--');
  document.getElementById('opt-ivrank').textContent    = (data.iv_rank || 0).toFixed(0) + '%';
  document.getElementById('pcr-badge').textContent     = 'PCR ' + pcr.toFixed(2);

  const ce = data.call_oi_total || 1, pe = data.put_oi_total || 1;
  document.getElementById('opt-oiratio').textContent = `${(ce/1e5).toFixed(1)}L : ${(pe/1e5).toFixed(1)}L`;

  // Support/Resistance
  const srEl = document.getElementById('sr-levels');
  const res = (data.resistance_levels || []).map(l =>
    `<div class="risk-row"><span style="color:var(--muted)">🔴 Resistance</span><span style="color:var(--red);font-weight:700">${fmtN(l)}</span></div>`
  ).join('');
  const sup = (data.support_levels || []).map(l =>
    `<div class="risk-row"><span style="color:var(--muted)">🟢 Support</span><span style="color:var(--green);font-weight:700">${fmtN(l)}</span></div>`
  ).join('');
  document.getElementById('opt-support').textContent = 'Support: ' + (data.support_levels||[]).slice(0,2).map(l=>fmtN(l)).join(' / ');
  srEl.innerHTML = res + sup || '<div style="color:var(--muted);text-align:center;padding:12px">No data</div>';

  // Options chain table
  const strikes = data.strikes || [];
  const atm     = data.atm_strike;
  const maxOI   = Math.max(...strikes.map(s => Math.max(s.ce_oi||0, s.pe_oi||0)), 1);

  const chainEl = document.getElementById('options-chain');
  chainEl.innerHTML = `
    <div style="padding:6px 4px;background:#0a0f16;font-weight:600;color:var(--muted);border-bottom:1px solid var(--border);text-align:right">CE OI | LTP | IV</div>
    <div style="padding:6px 4px;background:#0a0f16;font-weight:600;color:var(--muted);border-bottom:1px solid var(--border);text-align:center">Strike</div>
    <div style="padding:6px 4px;background:#0a0f16;font-weight:600;color:var(--muted);border-bottom:1px solid var(--border)">IV | LTP | PE OI</div>
    ` + strikes.map(s => {
      const isAtm = s.strike === atm;
      const ceBar = Math.round((s.ce_oi||0) / maxOI * 100);
      const peBar = Math.round((s.pe_oi||0) / maxOI * 100);
      const cls   = isAtm ? 'atm-row' : '';
      return `
      <div class="options-row ${cls}" style="display:contents">
        <div style="text-align:right;padding:5px 8px;border-bottom:1px solid #1c2128">
          <div class="ce-side">${(s.ce_iv||0).toFixed(1)}% | ${fmtN(s.ce_ltp||0)}</div>
          <div class="oi-bar"><div class="oi-bar-fill" style="width:${ceBar}%;background:var(--green);margin-left:auto"></div></div>
          <div style="color:var(--muted);font-size:10px">${((s.ce_oi||0)/1e5).toFixed(1)}L OI</div>
        </div>
        <div class="strike-col" style="padding:5px 4px;border-bottom:1px solid #1c2128;background:#0d1117;text-align:center;font-weight:700">
          ${fmtN(s.strike)}${isAtm ? ' ★' : ''}
        </div>
        <div style="padding:5px 8px;border-bottom:1px solid #1c2128">
          <div class="pe-side">${(s.pe_iv||0).toFixed(1)}% | ${fmtN(s.pe_ltp||0)}</div>
          <div class="oi-bar"><div class="oi-bar-fill" style="width:${peBar}%;background:var(--red)"></div></div>
          <div style="color:var(--muted);font-size:10px">${((s.pe_oi||0)/1e5).toFixed(1)}L OI</div>
        </div>
      </div>`;
    }).join('');
}

// ── WebSocket for live quotes ──────────────────────────────────────────
function connectWS() {
  const ws = new WebSocket(WS_URL);
  ws.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.type !== 'quotes') return;
      const q = data.quotes;
      updateTicker('nifty',     q['NIFTY']     || q['NIFTY 50']);
      updateTicker('banknifty', q['BANKNIFTY']  || q['NIFTY BANK']);
      updateTicker('finnifty',  q['FINNIFTY']);
      updateTicker('vix',       q['INDIA VIX']);
    } catch(e) {}
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
  ws.onerror = () => {};
}

function updateTicker(id, quote) {
  if (!quote) return;
  const priceEl  = document.getElementById(id + '-price');
  const changeEl = document.getElementById(id + '-change');
  if (!priceEl) return;
  const chg = quote.change_pct || 0;
  priceEl.textContent  = fmtN(quote.ltp || 0);
  priceEl.className    = 'ticker-price ' + (chg >= 0 ? 'up' : 'down');
  changeEl.textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%';
  changeEl.className   = 'ticker-change ' + (chg >= 0 ? 'up' : 'down');
}

// ── Load health/model status ───────────────────────────────────────────
async function loadHealth() {
  try {
    const res  = await fetch(`${API}/health`);
    const data = await res.json();
    const m    = data.model_metrics || {};
    document.getElementById('model-auc').textContent  = m.ensemble_auc || '--';
    document.getElementById('model-lstm').textContent = m.lstm_included ? '✅ Included' : '⚠️ RF+XGB only';
    document.getElementById('model-date').textContent = (m.trained_at||'').slice(0,16) || '--';
  } catch(e) {}
}

// ── Load Risk Status ───────────────────────────────────────────────────
async function loadRisk() {
  try {
    const res  = await fetch(`${API}/risk/status`);
    if (!res.ok) return;
    const s    = await res.json();
    const pnlEl = document.getElementById('risk-pnl');
    pnlEl.textContent  = (s.pnl_today >= 0 ? '+' : '') + fmt(s.pnl_today);
    pnlEl.style.color  = s.pnl_today >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('risk-deployed').textContent  = s.capital_deployed_pct + '%';
    document.getElementById('risk-drawdown').textContent  = s.drawdown_pct + '%';
    document.getElementById('risk-trades').textContent    = s.trades_today;
    document.getElementById('risk-positions').textContent = s.open_positions + ' / ' + s.max_positions;

    const halted = s.trading_halted || s.kill_switch;
    const statusEl = document.getElementById('risk-status');
    statusEl.textContent = halted ? (s.halt_reason || 'HALTED') : 'Active';
    statusEl.style.color = halted ? 'var(--red)' : 'var(--green)';

    const killBtn = document.getElementById('kill-btn');
    if (s.kill_switch) {
      killBtn.className = 'kill-switch kill-on';
      killBtn.textContent = '🛑 KILL SWITCH ON — Click to Resume';
    } else {
      killBtn.className = 'kill-switch kill-off';
      killBtn.textContent = '🟢 Trading Active — Click to Emergency Stop';
    }

    const deployPct = Math.min(100, s.capital_deployed_pct);
    document.getElementById('risk-deploy-bar').style.width = deployPct + '%';
    const lossPct = Math.min(100, Math.abs(s.pnl_today_pct) / 2 * 100);
    document.getElementById('risk-pnl-bar').style.width = lossPct + '%';
    document.getElementById('risk-pnl-bar').style.background =
      s.pnl_today >= 0 ? 'var(--green)' : 'var(--red)';
  } catch(e) {}
}

async function toggleKillSwitch() {
  try {
    const res  = await fetch(`${API}/risk/status`);
    const s    = await res.json();
    const endpoint = s.kill_switch ? `${API}/risk/kill-switch/deactivate` : `${API}/risk/kill-switch/activate`;
    await fetch(endpoint, {method:'POST'});
    await loadRisk();
  } catch(e) {}
}

// ── Symbol change handler ──────────────────────────────────────────────
document.getElementById('optSymbol').addEventListener('change', loadOptions);

// ── Init ───────────────────────────────────────────────────────────────
(async () => {
  await loadHealth();
  await loadSignals();
  await loadOptions();
  await loadRisk();
  connectWS();

  // Refresh intervals
  setInterval(loadSignals,  5 * 60 * 1000);   // signals every 5 min
  setInterval(loadOptions,  60 * 1000);        // options chain every 1 min
  setInterval(loadRisk,     30 * 1000);        // risk panel every 30 sec
  setInterval(loadHealth,   5 * 60 * 1000);   // model status every 5 min
})();
</script>
</body>
</html>
"""

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)
