# Monitoring

Each orchestrator generates a self-contained **HTML dashboard** in `logs/`:

```
logs/{device}_dashboard.html    # Auto-refreshes every 30 seconds
logs/{device}_status.json       # Machine-readable status
logs/{device}.log               # Plain-text rolling log
```

## Remote Monitoring

Serve the logs directory over HTTP:

```bash
cd logs && python -m http.server 8080
```

Then open `http://<device-ip>:8080/rtx5090_dashboard.html` in a browser.

## Dashboard Features

- Overall progress bar with run counts (completed / failed / skipped / pending)
- Phase indicator (export → inference → aggregation → complete)
- Filterable runs table with status, timing, and key metrics
- Recent log entries
- Auto-refresh every 30 seconds

## Log Files

For command-line monitoring:

```bash
tail -f logs/rtx5090.log
```

## GPU Monitoring

```bash
watch -n 2 nvidia-smi
```
