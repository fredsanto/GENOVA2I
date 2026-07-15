#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# tunnel_qwen.sh — SSH tunnel to ServerQwen running on HPC compute node.
#
# Run this on YOUR LAPTOP to connect to the cluster.
#
# Usage:
#   ./tunnel_qwen.sh                        # interactive (reads cluster path)
#   ./tunnel_qwen.sh --cluster-path /path   # auto: path to ServerQwen on cluster
#   ./tunnel_qwen.sh --node node01 --port 8002  # manual: known node + port
#
# Architecture:
#   Laptop ──SSH──> login node (curnagel.unil.ch) ──> compute node:PORT
#
# Prerequisites:
#   - SSH access to curnagel.unil.ch (password or key)
#   - ServerQwen running on the cluster (via sbatch launch_qwen.sh)
# ──────────────────────────────────────────────────────────────────────────────

LOGIN_NODE="curnagl.dcsr.unil.ch"
LOCAL_PORT=8002
REMOTE_PORT=""
REMOTE_NODE=""
CLUSTER_PATH=""

# Terminal colors
BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

usage() {
    echo "Usage: $(basename "$0") [--node NODE] [--port PORT] [--cluster-path PATH]"
    echo ""
    echo "  --node NODE         Compute node hostname (skip cluster lookup)"
    echo "  --port PORT         Server port on compute node (default: 8002)"
    echo "  --cluster-path PATH Path to ServerQwen dir on cluster"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node) REMOTE_NODE="$2"; shift 2 ;;
        --port) REMOTE_PORT="$2"; shift 2 ;;
        --cluster-path) CLUSTER_PATH="$2"; shift 2 ;;
        *) usage ;;
    esac
done

REMOTE_PORT="${REMOTE_PORT:-$LOCAL_PORT}"

echo -e "${BOLD}Qwen Variant Analysis — SSH Tunnel${NC}"
echo ""

# If node is known, skip cluster lookup
if [[ -z "$REMOTE_NODE" ]]; then
    # Determine cluster path
    if [[ -z "$CLUSTER_PATH" ]]; then
        echo -e "${DIM}Enter the path to ServerQwen on the cluster.${NC}"
        echo -e "${DIM}(Default: /work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/ServerQwen)${NC}"
        read -r -p "Path: " CLUSTER_PATH
        CLUSTER_PATH="${CLUSTER_PATH:-/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/ServerQwen}"
    fi

    echo -e "${YELLOW}✈ Looking up compute node from SLURM job...${NC}"

    # Fetch connection file from cluster
    CONN_DATA=$(ssh "fsantoni1@${LOGIN_NODE}" "cat \"${CLUSTER_PATH}/.connection\" 2>/dev/null" || true)

    if [[ -z "$CONN_DATA" ]]; then
        echo -e "${YELLOW}No .connection file found. Checking SLURM logs for running jobs...${NC}"
        JOB_INFO=$(ssh "fsantoni1@${LOGIN_NODE}" "cd \"${CLUSTER_PATH}\" && squeue --me --name=server_qwen --states=RUNNING -o '%A %N' -h 2>/dev/null | head -1" || true)
        if [[ -z "$JOB_INFO" ]]; then
            echo -e "${BOLD}No running server_qwen job found.${NC}"
            echo "1. SSH to the cluster: ssh ${LOGIN_NODE}"
            echo "2. cd ${CLUSTER_PATH}"
            echo "3. sbatch launch_qwen.sh"
            echo "4. Wait for the job to start, then re-run this script."
            exit 1
        fi
        REMOTE_NODE=$(echo "$JOB_INFO" | awk '{print $2}')
        JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
        echo -e "${GREEN}Found job $JOB_ID on node $REMOTE_NODE${NC}"
    else
        eval "$CONN_DATA"
        REMOTE_NODE="${SERVERQWN_NODE}"
        REMOTE_PORT="${SERVERQWN_PORT}"
        echo -e "${GREEN}Found connection: node=${REMOTE_NODE} port=${REMOTE_PORT}${NC}"
    fi

    # Verify server is responding
    SERVER_OK=$(ssh "fsantoni1@${LOGIN_NODE}" "curl -sf http://${REMOTE_NODE}:${REMOTE_PORT}/health 2>/dev/null" || echo "")
    if [[ -z "$SERVER_OK" ]]; then
        echo -e "${YELLOW}Warning: Server not responding yet. It may still be starting up.${NC}"
        echo -e "${YELLOW}Proceeding with tunnel anyway...${NC}"
    else
        echo -e "${GREEN}Server is running and healthy.${NC}"
    fi
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD} Setting up SSH tunnel...${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Local:     http://localhost:${LOCAL_PORT}"
echo "  Remote:    ${REMOTE_NODE}:${REMOTE_PORT}"
echo "  Login:     ${LOGIN_NODE}"
echo ""
echo -e "${DIM}Press Ctrl+C to close the tunnel.${NC}"
echo ""

ssh -N -L "${LOCAL_PORT}:${REMOTE_NODE}:${REMOTE_PORT}" "fsantoni1@${LOGIN_NODE}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3
