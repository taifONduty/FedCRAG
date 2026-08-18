#!/bin/bash
# Create the FedCRAG GPU VM (L4 24GB) on Google Cloud.
# Prereqs (see GCP_RUNBOOK.md): paid account, GPU quota >= 1, gcloud auth done.
# Usage:
#   bash gcp_provision.sh                       # on-demand L4
#   MODE=spot bash gcp_provision.sh             # spot (preemptible) L4
#   ZONE=us-east1-d NAME=fedcrag2 bash gcp_provision.sh
set -euo pipefail

ZONE=${ZONE:-us-central1-a}
NAME=${NAME:-fedcrag-l4}
MODE=${MODE:-standard}           # standard | spot
MACHINE=${MACHINE:-g2-standard-4}  # G2 machine types include one NVIDIA L4
DISK_GB=${DISK_GB:-150}

# Deep Learning VM image: NVIDIA driver + CUDA preinstalled (we bring our own
# python env via uv, so only the driver matters). If this family name has
# rotated, list current ones with:
#   gcloud compute images list --project deeplearning-platform-release \
#     --filter="family~'pytorch|common'" --sort-by=~creationTimestamp | head
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-latest-gpu}

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
    "${EXTRA[@]}"
set +x

echo ""
echo "Created. Next:  gcloud compute ssh $NAME --zone=$ZONE"
echo "REMEMBER:       gcloud compute instances stop $NAME --zone=$ZONE   (after every session)"
