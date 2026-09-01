# GCP runbook — FedCRAG campaign VM (verified 2026-08-18)

Target: one **NVIDIA L4 (24 GB)** VM — `g2-standard-4` (4 vCPU / 16 GB RAM,
L4 included in the machine type). ~$0.71/hr on-demand. Budget frame with $300
credit: W3′ campaign ≈ 30 GPU-h ≈ $21; full project through October ≈ 200–300
GPU-h ≈ $140–210. The credit covers it **only with stop-when-idle discipline**.

## Phase 0 — one-time console steps (only you can do these)

1. **Upgrade the Free Trial to a Paid account.** Verified 2026-08-18: trial
   accounts cannot attach GPUs or request GPU quota at all; upgrading keeps the
   unused $300 credit (within the original 90-day window). Console → Billing →
   "Activate full account".
2. **Request GPU quota** (after the upgrade): Console → IAM & Admin → Quotas →
   filter "GPUs (all regions)" → request limit **1**. Also check the per-region
   "NVIDIA L4 GPUs" quota in your chosen region (e.g. `us-central1`) is ≥ 1.
   Approval is usually minutes-to-hours for paid accounts.
3. Install the gcloud CLI locally and authenticate:
   `gcloud auth login && gcloud config set project <YOUR_PROJECT_ID>`

## Phase 1 — create the VM

```bash
bash gcp_provision.sh                 # on-demand L4 in us-central1-a
# or:  MODE=spot bash gcp_provision.sh     (cheaper; can be preempted mid-run)
# or:  ZONE=asia-southeast1-b bash gcp_provision.sh   (nearer region; check L4 quota there)
```

If creation fails with a resource-availability error, try another zone
(`us-central1-{a,b,c}`, `us-east1-d`, `europe-west4-a` all carry L4s).

## Phase 2 — bootstrap on the VM

```bash
gcloud compute ssh fedcrag-l4 --zone=us-central1-a
# first login: answer "y" if the image offers to install the NVIDIA driver
nvidia-smi                                  # must show L4, 24 GB
git clone <your-repo-url> FedCRAG && cd FedCRAG && git checkout w3-campaign
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest tests/ -q        # must be green
bash run_w3.sh smoke                        # minutes; sanity-check JSON + use_amp:true
```

Then the campaign, inside tmux so SSH drops don't kill runs:

```bash
tmux new -s w3
bash run_w3.sh controls          # then: .venv/bin/python check_headroom.py results/controls_contriever_seed42.json
PRIMARY=contriever bash run_w3.sh rt
PRIMARY=contriever bash run_w3.sh rs
# detach: Ctrl-b d   |   reattach: tmux attach -t w3
```

## Phase 3 — ship results + STOP THE METER

```bash
# on the VM: commit results to a branch (the May pattern, now with code pinned)
.venv/bin/pip freeze > requirements.lock
git checkout -b results/w3-contriever && git add results logs runs.tsv requirements.lock
git commit -m "results: W3 contriever campaign" && git push -u origin results/w3-contriever
# states_*.pt are gitignored — pull them separately when needed for mechanism work:
#   gcloud compute scp --recurse fedcrag-l4:~/FedCRAG/results ./results_vm --zone=us-central1-a
```

```bash
gcloud compute instances stop fedcrag-l4 --zone=us-central1-a    # after EVERY session
gcloud compute instances list                                     # verify TERMINATED
```

A stopped VM bills only its disk (~cents/day); a forgotten running VM bills
~$17/day. Set a budget alert: Console → Billing → Budgets & alerts → $50 / $150
/ $250 thresholds.

## Spot-mode caveat

Spot VMs can be preempted mid-run. Federated JSONs dump atomically per round,
so partial data survives, but re-run any interrupted arm from scratch
(optimizer state is not checkpointed). Use spot for `rt`/`controls` (short
runs), on-demand for the long `rs` matrix — or babysit spot with `tmux` +
re-queue. With this budget, on-demand for everything is also fine.
