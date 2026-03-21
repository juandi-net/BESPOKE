#!/bin/bash
# BESPOKE Nightly Pipeline
# Schedule via launchd or cron to run at 11 PM

set -eo pipefail

BESPOKE_DIR="$HOME/projects/bespoke"
VENV="$BESPOKE_DIR/.venv/bin/python"
LOG_DIR="$HOME/.bespoke/logs"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"

echo "=== BESPOKE Nightly Pipeline — $DATE ===" | tee -a "$LOG_DIR/$DATE.log"

# Stage 1: Capture new interactions
echo "[$(date)] Stage 1: Capture..." | tee -a "$LOG_DIR/$DATE.log"
$VENV -m bespoke.capture.pipeline 2>&1 | tee -a "$LOG_DIR/$DATE.log"

# Stage 2a: Extract and classify
echo "[$(date)] Stage 2a: Extract..." | tee -a "$LOG_DIR/$DATE.log"
$VENV -m bespoke.teach.stage2a 2>&1 | tee -a "$LOG_DIR/$DATE.log"

# Stage 3: Train (Claude Code would replace this in full autoresearch mode)
echo "[$(date)] Stage 3: Train..." | tee -a "$LOG_DIR/$DATE.log"
$VENV -m bespoke.train.train_sft --adapter-name general-v1 2>&1 | tee -a "$LOG_DIR/$DATE.log"

# Backup database
echo "[$(date)] Backing up database..." | tee -a "$LOG_DIR/$DATE.log"
cp "$HOME/.bespoke/bespoke.db" "$HOME/.bespoke/backups/bespoke-$DATE.db"

echo "[$(date)] Nightly pipeline complete." | tee -a "$LOG_DIR/$DATE.log"
