"""
app.py — Smart Energy RL  v2.0
================================
Flask API server implementing the full OpenEnv spec:
  POST /reset      → EnergyState
  POST /step       → StepResult
  GET  /state      → EnergyState
  GET  /health     → {"status": "ok"}
  GET  /tasks      → task list
  GET  /           → interactive dashboard

Launch: python app.py
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, render_template_string

try:
    from pydantic import ValidationError
    PYDANTIC = True
except ImportError:
    PYDANTIC = False

from env.energy_env import SmartEnergyEnv
from models.schemas import EnergyAction, EnergyState, StepResult
from tasks.task_definitions import TASKS, TaskStats

app = Flask(__name__)
env = SmartEnergyEnv()

# ─── Helper ──────────────────────────────────────────────────────

def state_to_dict(s) -> dict:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if hasattr(s, "to_dict"):
        return s.to_dict()
    return s.__dict__

def result_to_dict(r) -> dict:
    return {
        "state": state_to_dict(r.state),
        "reward": r.reward,
        "done": r.done,
        "info": r.info,
    }

# ─── OpenEnv Core Endpoints ──────────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset():
    """Reset the environment. Returns initial EnergyState."""
    seed = request.get_json(silent=True) or {}
    if "seed" in seed:
        random.seed(int(seed["seed"]))
    state = env.reset()
    return jsonify(state_to_dict(state)), 200

@app.route("/step", methods=["POST"])
def step():
    """Advance one timestep. Body: {action: {hvac_setpoint, battery_action}}"""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON with 'action' key."}), 400

    raw_action = body.get("action", body)  # accept flat or nested
    try:
        if PYDANTIC:
            action = EnergyAction(**raw_action)
        else:
            action = EnergyAction(
                hvac_setpoint=float(raw_action.get("hvac_setpoint", 0.5)),
                battery_action=float(raw_action.get("battery_action", 0.0)),
            )
    except Exception as e:
        return jsonify({"error": f"Invalid action: {e}"}), 422

    result = env.step(action)
    return jsonify(result_to_dict(result)), 200

@app.route("/state", methods=["GET"])
def get_state():
    """Return current state without advancing time."""
    s = env.state()
    if s is None:
        env.reset()
        s = env.state()
    return jsonify(state_to_dict(s)), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "2.0.0"}), 200

@app.route("/tasks", methods=["GET"])
def list_tasks():
    return jsonify([
        {"id": tid, "name": t.name, "difficulty": t.difficulty}
        for tid, t in TASKS.items()
    ]), 200

# ─── Interactive Dashboard ───────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Energy RL — Dashboard v2.0</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  header { background: linear-gradient(135deg, #1e3a8a, #0f766e); padding: 20px 32px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.5rem; font-weight: 700; }
  header span { font-size: 0.85rem; opacity: 0.7; }
  .container { max-width: 1280px; margin: 0 auto; padding: 24px; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; align-items: center; }
  select, button { padding: 8px 16px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #e2e8f0; font-size: 0.9rem; cursor: pointer; }
  button.primary { background: #0f766e; border-color: #0f766e; font-weight: 600; }
  button.primary:hover { background: #0d9488; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 12px; margin-bottom: 24px; }
  .kpi { background: #1e293b; border-radius: 10px; padding: 14px 16px; border-left: 3px solid #0f766e; }
  .kpi label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; opacity: .6; }
  .kpi value { display: block; font-size: 1.4rem; font-weight: 700; margin-top: 4px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #1e293b; border-radius: 12px; padding: 16px; }
  .card h3 { font-size: 0.85rem; opacity: .7; margin-bottom: 12px; text-transform: uppercase; letter-spacing: .05em; }
  canvas { max-height: 220px; }
  .zone-bar { display: flex; gap: 8px; margin-top: 8px; }
  .zone { flex: 1; padding: 8px; border-radius: 6px; font-size: 0.8rem; text-align: center; }
  .zone.ok { background: #064e3b; border: 1px solid #10b981; }
  .zone.warn { background: #7f1d1d; border: 1px solid #ef4444; }
  #log { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.75rem; max-height: 160px; overflow-y: auto; margin-top: 16px; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div>⚡</div>
  <div><h1>Smart Energy RL — v2.0</h1><span>Dynamic Pricing · Battery Degradation · Multi-Room HVAC · Weather Simulation</span></div>
</header>
<div class="container">
  <div class="controls">
    <select id="taskSel">
      <option value="task1_peak_shaving">Task 1 — Peak Shaving (Easy)</option>
      <option value="task2_comfort_first">Task 2 — Comfort First (Medium)</option>
      <option value="task3_solar_arbitrage">Task 3 — Solar Arbitrage (Hard)</option>
    </select>
    <select id="agentSel">
      <option value="rule_based">Rule-Based Agent</option>
      <option value="solar_arbitrage">Solar Arbitrage Agent</option>
      <option value="random">Random Agent</option>
    </select>
    <button class="primary" id="runBtn" onclick="runSimulation()">▶ Run Episode</button>
    <span id="status" style="opacity:.6; font-size:.85rem;"></span>
  </div>
  <div class="kpi-grid">
    <div class="kpi"><label>Total Cost</label><value id="kCost">—</value></div>
    <div class="kpi"><label>Score</label><value id="kScore">—</value></div>
    <div class="kpi"><label>Violations</label><value id="kViol">—</value></div>
    <div class="kpi"><label>Solar Sold</label><value id="kSold">—</value></div>
    <div class="kpi"><label>Battery Health</label><value id="kHealth">—</value></div>
    <div class="kpi"><label>Step</label><value id="kStep">0/24</value></div>
  </div>
  <div id="zoneBar" class="zone-bar"></div>
  <div class="charts">
    <div class="card"><h3>Energy Flow (kW)</h3><canvas id="cEnergy"></canvas></div>
    <div class="card"><h3>Temperature (°C)</h3><canvas id="cTemp"></canvas></div>
    <div class="card"><h3>Battery (kWh) &amp; Health</h3><canvas id="cBatt"></canvas></div>
    <div class="card"><h3>Hourly Cost ($) &amp; Grid Price</h3><canvas id="cCost"></canvas></div>
  </div>
  <div id="log"></div>
</div>
<script>
let charts = {};
function initCharts() {
  const opts = (labels, datasets) => ({ type:'line', data:{labels,datasets}, options:{responsive:true, animation:false, plugins:{legend:{labels:{color:'#94a3b8',boxWidth:10,font:{size:11}}}}, scales:{x:{ticks:{color:'#64748b',maxTicksLimit:12}},y:{ticks:{color:'#64748b'}}}}});
  const mk = (id, labels, datasets) => {
    if(charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id).getContext('2d'), opts(labels, datasets));
  };
  const hrs = Array.from({length:25},(_,i)=>i+'h');
  const ds = (label, color, data=[]) => ({label, borderColor:color, backgroundColor:color+'22', data, tension:.3, pointRadius:2, fill:true});
  mk('cEnergy', hrs, [ds('Solar','#f59e0b'), ds('HVAC','#ef4444'), ds('Grid','#3b82f6')]);
  mk('cTemp',   hrs, [ds('Indoor','#10b981'), ds('Outdoor','#f97316'), ds('Server Room','#a855f7')]);
  mk('cBatt',   hrs, [ds('Battery kWh','#6366f1'), ds('Health %×10','#84cc16')]);
  mk('cCost',   hrs, [ds('Hourly Cost $×10','#f43f5e'), ds('Grid Price $×10','#fb923c')]);
}
function push(id, idx, ...vals) {
  const c = charts[id];
  vals.forEach((v,i) => c.data.datasets[i].data[idx] = v);
  c.update('none');
}
function log(msg) {
  const el = document.getElementById('log');
  el.innerHTML += msg + '<br>';
  el.scrollTop = el.scrollHeight;
}
function agent(state, task) {
  const bp = state.battery_level / state.battery_capacity;
  let bat=0, hvac=0.4;
  if (task==='task1_peak_shaving') {
    bat = state.is_peak_hour ? (bp>0.3?-0.8:0) : (state.solar_power>0.5?0.9:0.3);
    hvac = state.is_peak_hour ? 0.3 : 0.5;
  } else if (task==='task2_comfort_first') {
    const err = Math.abs(state.indoor_temp-22);
    hvac = err<1?0.2:err<3?0.5:0.8;
    bat = state.is_peak_hour ? (bp>0.3?-0.7:0) : (state.solar_power>1?0.8:0.2);
  } else {
    const h = state.hour;
    bat = (h>=10&&h<=15&&state.solar_power>0.5)?1.0:(state.is_peak_hour&&bp>0.2?-1.0:0);
    const err = Math.abs(state.indoor_temp-22);
    hvac = state.is_peak_hour?(err<2?0.1:0.3):(err<1.5?0.3:0.6);
  }
  return {hvac_setpoint: Math.max(0,Math.min(1,hvac)), battery_action: Math.max(-1,Math.min(1,bat))};
}
async function runSimulation() {
  const taskId = document.getElementById('taskSel').value;
  document.getElementById('runBtn').disabled = true;
  document.getElementById('log').innerHTML = '';
  document.getElementById('status').textContent = 'Resetting...';
  initCharts();
  const rr = await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  let state = await rr.json();
  let score=0, solarSold=0;
  document.getElementById('status').textContent = 'Running...';
  for (let step=0; step<24; step++) {
    const action = agent(state, taskId);
    const sr = await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
    const res = await sr.json();
    const info = res.info; state = res.state;
    solarSold = info.total_solar_sold_back_kwh||0;
    push('cEnergy', step, info.solar_power_kw, info.hvac_kw, Math.max(0,info.net_from_grid_kw));
    push('cTemp',   step, state.indoor_temp, info.outdoor_temp, (state.zone_temps||{}).server_room||state.indoor_temp+1);
    push('cBatt',   step, state.battery_level, state.battery_health*10);
    push('cCost',   step, (info.hourly_cost||0)*10, info.grid_price*10);
    document.getElementById('kStep').textContent = (step+1)+'/24';
    document.getElementById('kCost').textContent = '$'+(state.total_cost).toFixed(3);
    document.getElementById('kViol').textContent = state.comfort_violations;
    document.getElementById('kSold').textContent = solarSold.toFixed(2)+' kWh';
    document.getElementById('kHealth').textContent = (state.battery_health*100).toFixed(1)+'%';
    // Zone temps
    if (state.zone_temps) {
      document.getElementById('zoneBar').innerHTML = Object.entries(state.zone_temps).map(([z,t])=>
        `<div class="zone ${t>=20&&t<=26?'ok':'warn'}">${z}<br><b>${t.toFixed(1)}°C</b></div>`).join('');
    }
    log(`[Step ${step+1}] hvac=${action.hvac_setpoint.toFixed(2)} bat=${action.battery_action.toFixed(2)} | temp=${state.indoor_temp.toFixed(1)}°C solar=${info.solar_power_kw.toFixed(2)}kW cost=$${(info.hourly_cost||0).toFixed(4)} reward=${res.reward.toFixed(4)}`);
    await new Promise(r=>setTimeout(r,40));
  }
  // Grading
  const gr = await fetch('/grade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId, total_cost:state.total_cost, comfort_violations:state.comfort_violations, solar_generated:state.solar_power||0, solar_sold_back:solarSold, total_reward:0, steps:24})});
  if (gr.ok) { const gj = await gr.json(); score = gj.score||0; }
  document.getElementById('kScore').textContent = score.toFixed(4);
  document.getElementById('status').textContent = `Done! Score: ${score.toFixed(4)}`;
  document.getElementById('runBtn').disabled = false;
  log(`[DONE] cost=$${state.total_cost.toFixed(4)} violations=${state.comfort_violations} sold_back=${solarSold.toFixed(2)}kWh → score=${score.toFixed(4)}`);
}
initCharts();
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/grade", methods=["POST"])
def grade():
    """Grade an episode. Body: {task_id, total_cost, comfort_violations, solar_generated, solar_sold_back, total_reward, steps}"""
    body = request.get_json(force=True, silent=True) or {}
    task_id = body.get("task_id", "task1_peak_shaving")
    if task_id not in TASKS:
        return jsonify({"error": f"Unknown task_id: {task_id}"}), 404
    try:
        stats = TaskStats(
            total_cost=float(body.get("total_cost", 0)),
            comfort_violations=int(body.get("comfort_violations", 0)),
            solar_generated=float(body.get("solar_generated", 0)),
            solar_sold_back=float(body.get("solar_sold_back", 0)),
            total_reward=float(body.get("total_reward", 0)),
            steps=int(body.get("steps", 24)),
        )
        score = TASKS[task_id].grade(stats)
        return jsonify({"task_id": task_id, "score": score}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 422

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
