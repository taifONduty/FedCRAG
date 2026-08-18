#!/bin/bash
# Create the FedCRAG GPU VM (L4 24GB) on Google Cloud.
# Prereqs (see GCP_RUNBOOK.md): paid account, GPU quota >= 1, gcloud auth done.
# Usage:
#   bash gcp_provision.sh                       # on-demand L4
#   MODE=spot bash gcp_provision.sh             # spot (preemptible) L4
#   ZONE=us-east1-d NAME=fedcrag2 bash gcp_provision.sh
set -euo pipefail

# Defaults match the student's previously WORKING L4 creation (asia-south1-c,
# 2026; account project rokkh-503122) — closest region to Dhaka, quota proven.
ZONE=${ZONE:-asia-south1-c}
NAME=${NAME:-thesis-fedcrag}
MODE=${MODE:-standard}           # standard | spot
MACHINE=${MACHINE:-g2-standard-8}  # G2 machine types include one NVIDIA L4 24GB
DISK_GB=${DISK_GB:-200}

# Deep Learning VM image: NVIDIA driver + CUDA preinstalled (we bring our own
# python env via uv, so only the driver matters). This family is the one the
# student already used successfully. If the name rotates, list current ones:
#   gcloud compute images list --project deeplearning-platform-release \
#     --filter="family~'pytorch|common'" --sort-by=~creationTimestamp | head
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}

EXTRA=()
if [ "$MODE" = "spot" ]; then
    EXTRA+=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

set -x
gcloud compute instances create "$NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    "${EXTRA[@]}"
set +x

echo ""
echo "Created. Next:  gcloud compute ssh $NAME --zone=$ZONE"
echo "REMEMBER:       gcloud compute instances stop $NAME --zone=$ZONE   (after every session)"
