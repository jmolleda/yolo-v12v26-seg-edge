#!/bin/bash
# Auto-rebuild and push results dashboard to gh-pages branch.
# Run in a tmux pane on the RTX workstation while the benchmark runs.
#
# Usage:
#   bash scripts/autopush_dashboard.sh          # default: every 10 minutes
#   bash scripts/autopush_dashboard.sh 300      # every 5 minutes

INTERVAL=${1:-600}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKTREE_DIR=$(mktemp -d)
DASHBOARD_FILE="$REPO_DIR/docs/results_dashboard.html"
LAST_HASH=""

cleanup() {
    echo "[$(date '+%H:%M:%S')] Cleaning up worktree..."
    cd "$REPO_DIR"
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null
    exit 0
}
trap cleanup EXIT INT TERM

echo "=== Dashboard autopush ==="
echo "Repo:     $REPO_DIR"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

# Set up a persistent worktree for gh-pages
cd "$REPO_DIR"
git fetch origin gh-pages:gh-pages 2>/dev/null
git worktree add "$WORKTREE_DIR" gh-pages

while true; do
    echo "[$(date '+%H:%M:%S')] Rebuilding dashboard..."
    cd "$REPO_DIR"
    python build_results_dashboard.py > /dev/null 2>&1

    if [ ! -f "$DASHBOARD_FILE" ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: Dashboard not generated"
        sleep "$INTERVAL"
        continue
    fi

    # Check if any completed run reports changed (report.txt only — results.csv changes every epoch)
    REPORT_FILES=$(find "$REPO_DIR"/run_rtx5090/results -name "report.txt" 2>/dev/null)
    if [ -z "$REPORT_FILES" ]; then
        echo "[$(date '+%H:%M:%S')] No completed runs yet, skipping push"
        sleep "$INTERVAL"
        continue
    fi
    CURRENT_FINGERPRINT=$(echo "$REPORT_FILES" | sort | xargs ls -l 2>/dev/null | md5sum)
    if [ "$CURRENT_FINGERPRINT" = "$LAST_HASH" ]; then
        echo "[$(date '+%H:%M:%S')] No new data, skipping push"
    else
        cp "$DASHBOARD_FILE" "$WORKTREE_DIR/results_dashboard.html"
        cd "$WORKTREE_DIR"
        git add results_dashboard.html
        git commit -m "Auto-update results dashboard [$(date '+%Y-%m-%d %H:%M')]"
        git push origin gh-pages
        LAST_HASH="$CURRENT_FINGERPRINT"
        echo "[$(date '+%H:%M:%S')] Pushed update to gh-pages"
    fi

    echo "[$(date '+%H:%M:%S')] Next update in ${INTERVAL}s..."
    sleep "$INTERVAL"
done
