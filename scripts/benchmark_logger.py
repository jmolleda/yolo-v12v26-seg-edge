"""Benchmark logging system with JSON status and HTML dashboard.

Provides a BenchmarkLogger class that tracks all runs and generates:
- A JSON status file (single source of truth)
- A self-contained HTML dashboard (auto-refreshes every 30s)
- A plain-text append log (for tail -f)
"""

import datetime
import json
import os
import platform

LOG_BUFFER_SIZE = 50


class BenchmarkLogger:
    """Tracks benchmark progress and generates live dashboard."""

    def __init__(self, device, logs_dir=None):
        self.device = device
        self.machine = platform.node()
        self.started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if logs_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logs_dir = os.path.join(project_root, "logs")
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        self.json_path = os.path.join(self.logs_dir, f"{device}_status.json")
        self.html_path = os.path.join(self.logs_dir, f"{device}_dashboard.html")
        self.log_path = os.path.join(self.logs_dir, f"{device}.log")

        self.state = {
            "device": device,
            "machine": self.machine,
            "started_at": self.started_at,
            "updated_at": self.started_at,
            "phase": "initializing",
            "current_run": None,
            "counters": {
                "train": {"completed": 0, "skipped": 0, "failed": 0, "total": 0},
                "export": {"completed": 0, "skipped": 0, "failed": 0, "total": 0},
                "infer": {"completed": 0, "skipped": 0, "failed": 0, "total": 0},
            },
            "runs": [],
            "log": [],
        }

    def register_runs(self, runs):
        """Pre-populate all runs as pending.

        Args:
            runs: List of run dicts from load_experiments().
        """
        for run in runs:
            run_id = self._make_run_id(run)
            action = run.get("action", "infer")
            # Determine category
            if "train" in action:
                category = "train"
            elif "export" in action:
                category = "export"
            else:
                category = "infer"

            self.state["runs"].append({
                "id": run_id,
                "experiment": run.get("experiment_name", ""),
                "architecture": run.get("architecture", ""),
                "model_size": run.get("model_size", ""),
                "task": run.get("task", ""),
                "approach": run.get("approach", ""),
                "format": run.get("format", ""),
                "precision": run.get("precision", ""),
                "imgsz": run.get("imgsz", 640),
                "batch": run.get("batch", 1) if category != "train" else None,
                "action": action,
                "category": category,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration": None,
                "result": None,
                "error": None,
            })

        # Count totals per category
        for run_entry in self.state["runs"]:
            cat = run_entry["category"]
            if cat in self.state["counters"]:
                self.state["counters"][cat]["total"] += 1

        self.log("info", f"Registered {len(runs)} runs")
        self._flush()

    def set_phase(self, phase):
        """Update current phase label."""
        self.state["phase"] = phase
        self.log("info", f"Phase: {phase}")
        self._flush()

    def start_run(self, run_id):
        """Mark a run as running."""
        run_entry = self._find_run(run_id)
        if not run_entry:
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_entry["status"] = "running"
        run_entry["started_at"] = now

        idx = self._run_index(run_id)
        total = len(self.state["runs"])
        desc = self._run_description(run_entry)
        self.state["current_run"] = {
            "index": idx + 1,
            "total": total,
            "description": desc,
            "started_at": now,
        }

        self.log("info", f"[{idx + 1}/{total}] Started: {desc}")
        self._flush()

    def complete_run(self, run_id, result=None):
        """Mark a run as done with optional results."""
        run_entry = self._find_run(run_id)
        if not run_entry:
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_entry["status"] = "done"
        run_entry["finished_at"] = now
        run_entry["result"] = result

        # Update batch with actual value if provided (AutoBatch resolves -1)
        if result and result.get("batch") is not None:
            run_entry["batch"] = result["batch"]

        if run_entry["started_at"]:
            start = datetime.datetime.strptime(run_entry["started_at"], "%Y-%m-%d %H:%M:%S")
            end = datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
            run_entry["duration"] = str(end - start).split(".")[0]

        cat = run_entry["category"]
        if cat in self.state["counters"]:
            self.state["counters"][cat]["completed"] += 1

        self.state["current_run"] = None
        desc = self._run_description(run_entry)
        if result and result.get("fps"):
            self.log("info", f"Completed: {desc} ({result['fps']:.1f} FPS)")
        else:
            self.log("info", f"Completed: {desc}")
        self._flush()

    def fail_run(self, run_id, error):
        """Mark a run as failed."""
        run_entry = self._find_run(run_id)
        if not run_entry:
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_entry["status"] = "failed"
        run_entry["finished_at"] = now
        run_entry["error"] = str(error)

        if run_entry["started_at"]:
            start = datetime.datetime.strptime(run_entry["started_at"], "%Y-%m-%d %H:%M:%S")
            end = datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
            run_entry["duration"] = str(end - start).split(".")[0]

        cat = run_entry["category"]
        if cat in self.state["counters"]:
            self.state["counters"][cat]["failed"] += 1

        self.state["current_run"] = None
        desc = self._run_description(run_entry)
        self.log("error", f"Failed: {desc} - {error}")
        self._flush()

    def skip_run(self, run_id, reason="already done", report_path=None):
        """Mark a run as skipped, loading metrics from existing report if available."""
        run_entry = self._find_run(run_id)
        if not run_entry:
            return
        run_entry["status"] = "skipped"
        run_entry["error"] = reason

        # Load metrics from existing report
        if report_path and os.path.exists(report_path):
            self._load_report_into_run(run_entry, report_path)

        cat = run_entry["category"]
        if cat in self.state["counters"]:
            self.state["counters"][cat]["skipped"] += 1

        desc = self._run_description(run_entry)
        self.log("info", f"Skipped: {desc} ({reason})")
        self._flush()

    @staticmethod
    def _load_report_into_run(run_entry, report_path):
        """Parse report.txt and populate run_entry with saved metrics."""
        import re
        with open(report_path, "r") as f:
            content = f.read()

        def _extract(pattern, text, cast=str):
            m = re.search(pattern, text)
            return cast(m.group(1)) if m else None

        run_entry["duration"] = _extract(r"Duration:\s+(.+)", content)
        batch = _extract(r"Batch size:\s+(\d+)", content, int)
        if batch is not None:
            run_entry["batch"] = batch

        result = {}
        fps = _extract(r"FPS:\s+([\d.]+)", content, float)
        if fps:
            result["fps"] = fps
        mAP50 = _extract(r"mAP50:\s+([\d.]+)", content, float)
        if mAP50 is not None:
            result["mAP50"] = mAP50
        mAP50_95 = _extract(r"mAP50-95:\s+([\d.]+)", content, float)
        if mAP50_95 is not None:
            result["mAP50_95"] = mAP50_95

        start = _extract(r"Start time:\s+(.+)", content)
        end = _extract(r"End time:\s+(.+)", content)
        if start:
            run_entry["started_at"] = start.strip()
        if end:
            run_entry["finished_at"] = end.strip()

        if result:
            run_entry["result"] = result

    def log(self, level, message):
        """Add a message to the rolling log buffer and plain-text log."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"ts": now, "level": level, "msg": message}
        self.state["log"].append(entry)
        if len(self.state["log"]) > LOG_BUFFER_SIZE:
            self.state["log"] = self.state["log"][-LOG_BUFFER_SIZE:]

        # Append to plain-text log
        log_line = f"[{now}] [{level.upper()}] {message}\n"
        print(log_line.rstrip())
        try:
            with open(self.log_path, "a") as f:
                f.write(log_line)
        except Exception:
            pass

    # -- Internal methods --

    @staticmethod
    def make_run_id(run):
        """Generate a deterministic run ID from run parameters. Public static method."""
        return BenchmarkLogger._make_run_id(run)

    @staticmethod
    def _make_run_id(run):
        task_key = "seg" if run.get("task") == "segment" else "det"
        return (
            f"{run.get('experiment_name', '')}_"
            f"{run.get('architecture', '')}_{task_key}_"
            f"{run.get('model_size', '')}_{run.get('approach', '')}_"
            f"{run.get('format', '')}_{run.get('precision', '')}_"
            f"img{run.get('imgsz', 640)}_b{run.get('batch', 1)}"
        )

    def _find_run(self, run_id):
        for r in self.state["runs"]:
            if r["id"] == run_id:
                return r
        return None

    def _run_index(self, run_id):
        for i, r in enumerate(self.state["runs"]):
            if r["id"] == run_id:
                return i
        return 0

    @staticmethod
    def _run_description(run_entry):
        batch = run_entry['batch'] if run_entry['batch'] is not None else '-'
        return (
            f"{run_entry['architecture']} {run_entry['model_size']} | "
            f"{run_entry['format']} {run_entry['precision']} | "
            f"{run_entry['task']} | {run_entry['approach']} | "
            f"img={run_entry['imgsz']} b={batch}"
        )

    def _flush(self):
        """Write JSON status and regenerate HTML dashboard."""
        self.state["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Atomic JSON write
        tmp_path = self.json_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp_path, self.json_path)
        except Exception as e:
            print(f"Warning: could not write status JSON: {e}")

        # Generate HTML dashboard
        try:
            self._generate_html()
        except Exception as e:
            print(f"Warning: could not generate HTML dashboard: {e}")

    def _generate_html(self):
        """Generate a self-contained HTML dashboard with embedded JSON."""
        data_json = json.dumps(self.state)
        html = _HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/", data_json)
        tmp_path = self.html_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(html)
        os.replace(tmp_path, self.html_path)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center;
            padding: 20px; background: #1e293b; border-radius: 12px; margin-bottom: 20px; }
  .header h1 { font-size: 1.5em; color: #38bdf8; }
  .header .info { text-align: right; color: #94a3b8; font-size: 0.9em; }
  .phase-badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
                 font-weight: 600; font-size: 0.85em; text-transform: uppercase; }
  .phase-training { background: #7c3aed; color: white; }
  .phase-export { background: #d97706; color: white; }
  .phase-inference { background: #2563eb; color: white; }
  .phase-aggregation { background: #059669; color: white; }
  .phase-complete { background: #16a34a; color: white; }
  .phase-failed { background: #dc2626; color: white; }
  .phase-initializing { background: #475569; color: white; }

  .progress-container { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .progress-bar { width: 100%; height: 28px; background: #334155; border-radius: 14px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 14px; transition: width 0.5s;
                   background: linear-gradient(90deg, #2563eb, #38bdf8); }
  .progress-text { text-align: center; margin-top: 8px; color: #94a3b8; }

  .counters { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 12px; margin-bottom: 20px; }
  .counter-card { background: #1e293b; border-radius: 12px; padding: 16px; text-align: center; }
  .counter-card h3 { color: #94a3b8; font-size: 0.8em; text-transform: uppercase; margin-bottom: 8px; }
  .counter-values { display: flex; justify-content: space-around; }
  .counter-val { text-align: center; }
  .counter-val .num { font-size: 1.5em; font-weight: 700; }
  .counter-val .label { font-size: 0.7em; color: #64748b; }
  .c-done { color: #4ade80; }
  .c-skip { color: #fbbf24; }
  .c-fail { color: #f87171; }
  .c-total { color: #94a3b8; }

  .current-run { background: #1e3a5f; border: 1px solid #2563eb; border-radius: 12px;
                 padding: 16px; margin-bottom: 20px; }
  .current-run.idle { background: #1e293b; border-color: #334155; }
  .current-run h3 { color: #38bdf8; font-size: 0.9em; margin-bottom: 8px; }
  .current-run .desc { font-size: 1.1em; font-weight: 600; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
  .running-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                       background: #38bdf8; animation: pulse 1.5s infinite; margin-right: 8px; }

  .runs-table { background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 20px;
                overflow-x: auto; }
  .filters { margin-bottom: 12px; display: flex; gap: 12px; flex-wrap: wrap; }
  .filters label { cursor: pointer; font-size: 0.85em; color: #94a3b8; }
  .filters input { margin-right: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th { text-align: left; padding: 8px 6px; color: #64748b; border-bottom: 1px solid #334155;
       font-size: 0.75em; text-transform: uppercase; }
  td { padding: 6px; border-bottom: 1px solid #1e293b; }
  tr:hover { background: #334155; }
  .status { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: 600; }
  .s-pending { background: #334155; color: #94a3b8; }
  .s-running { background: #1e3a5f; color: #38bdf8; }
  .s-done { background: #14532d; color: #4ade80; }
  .s-failed { background: #450a0a; color: #f87171; }
  .s-skipped { background: #422006; color: #fbbf24; }

  .log-section { background: #1e293b; border-radius: 12px; padding: 16px; }
  .log-section h3 { color: #94a3b8; font-size: 0.9em; margin-bottom: 12px; }
  .log-entry { font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.8em;
               padding: 3px 0; border-bottom: 1px solid #0f172a; }
  .log-ts { color: #64748b; }
  .log-info { color: #94a3b8; }
  .log-error { color: #f87171; }
  .log-warn { color: #fbbf24; }
</style>
</head>
<body>
<div id="app"></div>
<script>
const DATA = /*DATA_PLACEHOLDER*/;

function render() {
  if (!DATA) { document.getElementById('app').innerHTML = '<p>No data</p>'; return; }
  const d = DATA;
  const total = d.runs.length;
  const done = d.runs.filter(r => r.status === 'done').length;
  const failed = d.runs.filter(r => r.status === 'failed').length;
  const skipped = d.runs.filter(r => r.status === 'skipped').length;
  const progress = total > 0 ? ((done + failed + skipped) / total * 100).toFixed(1) : 0;

  let html = '';

  // Header
  html += `<div class="header">
    <div>
      <h1>${d.device.toUpperCase()} Benchmark</h1>
      <span class="phase-badge phase-${d.phase}">${d.phase}</span>
    </div>
    <div class="info">
      <div>Machine: <strong>${d.machine}</strong></div>
      <div>Started: ${d.started_at}</div>
      <div>Updated: ${d.updated_at}</div>
    </div>
  </div>`;

  // Progress bar
  html += `<div class="progress-container">
    <div class="progress-bar"><div class="progress-fill" style="width:${progress}%"></div></div>
    <div class="progress-text">${done + failed + skipped} / ${total} runs (${progress}%) &mdash;
      <span class="c-done">${done} done</span> &middot;
      <span class="c-skip">${skipped} skipped</span> &middot;
      <span class="c-fail">${failed} failed</span></div>
  </div>`;

  // Counters
  html += '<div class="counters">';
  for (const [cat, c] of Object.entries(d.counters)) {
    if (c.total === 0) continue;
    html += `<div class="counter-card">
      <h3>${cat}</h3>
      <div class="counter-values">
        <div class="counter-val"><div class="num c-done">${c.completed}</div><div class="label">Done</div></div>
        <div class="counter-val"><div class="num c-skip">${c.skipped}</div><div class="label">Skip</div></div>
        <div class="counter-val"><div class="num c-fail">${c.failed}</div><div class="label">Fail</div></div>
        <div class="counter-val"><div class="num c-total">${c.total}</div><div class="label">Total</div></div>
      </div>
    </div>`;
  }
  html += '</div>';

  // Current run
  if (d.current_run) {
    html += `<div class="current-run">
      <h3><span class="running-indicator"></span>Running (${d.current_run.index}/${d.current_run.total})</h3>
      <div class="desc">${d.current_run.description}</div>
      <div style="color:#64748b;margin-top:4px">Started: ${d.current_run.started_at}</div>
    </div>`;
  } else {
    html += `<div class="current-run idle"><h3>No run in progress</h3></div>`;
  }

  // Filters + Runs table
  html += `<div class="runs-table">
    <div class="filters">
      <label><input type="checkbox" checked onchange="toggleStatus('pending')"> Pending</label>
      <label><input type="checkbox" checked onchange="toggleStatus('running')"> Running</label>
      <label><input type="checkbox" checked onchange="toggleStatus('done')"> Done</label>
      <label><input type="checkbox" checked onchange="toggleStatus('failed')"> Failed</label>
      <label><input type="checkbox" checked onchange="toggleStatus('skipped')"> Skipped</label>
    </div>
    <table><thead><tr>
      <th>Status</th><th>Exp</th><th>Action</th><th>Arch</th><th>Size</th><th>Task</th>
      <th>Approach</th><th>Format</th><th>Prec</th><th>ImgSz</th><th>Batch</th>
      <th>Duration</th><th>FPS</th><th>mAP50</th>
    </tr></thead><tbody>`;

  for (const r of d.runs) {
    const fps = r.result && r.result.fps ? r.result.fps.toFixed(1) : '-';
    const map50 = r.result && r.result.map50 ? r.result.map50.toFixed(4) : '-';
    const batch = r.batch != null ? r.batch : '-';
    const err = r.error && r.status === 'failed' ? ` title="${r.error}"` : '';
    html += `<tr class="run-row" data-status="${r.status}">
      <td><span class="status s-${r.status}"${err}>${r.status}</span></td>
      <td>${r.experiment}</td><td>${r.category}</td><td>${r.architecture}</td><td>${r.model_size}</td>
      <td>${r.task}</td><td>${r.approach}</td><td>${r.format}</td><td>${r.precision}</td>
      <td>${r.imgsz}</td><td>${batch}</td>
      <td>${r.duration || '-'}</td><td>${fps}</td><td>${map50}</td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  // Log
  html += '<div class="log-section"><h3>Recent Log</h3>';
  const logs = d.log.slice().reverse().slice(0, 20);
  for (const e of logs) {
    const cls = e.level === 'error' ? 'log-error' : e.level === 'warn' ? 'log-warn' : 'log-info';
    html += `<div class="log-entry"><span class="log-ts">${e.ts}</span>
      <span class="${cls}"> [${e.level.toUpperCase()}] ${e.msg}</span></div>`;
  }
  html += '</div>';

  document.getElementById('app').innerHTML = html;
}

const hiddenStatuses = new Set();
function toggleStatus(status) {
  if (hiddenStatuses.has(status)) hiddenStatuses.delete(status);
  else hiddenStatuses.add(status);
  document.querySelectorAll('.run-row').forEach(row => {
    row.style.display = hiddenStatuses.has(row.dataset.status) ? 'none' : '';
  });
}

render();
</script>
</body>
</html>"""
