#!/usr/bin/env bash
set -euo pipefail

# Deploy the training job to the WDC cluster.
# Run from your laptop (inside erebus-v2 dir):
#   1. First: bash scripts/build_and_push.sh  (builds + pushes container)
#   2. Then:  bash scripts/deploy.sh          (creates PVC + launches job)

BASTION="soyr-redhat@169.62.21.215"

echo "==> Creating PVC..."
ssh "${BASTION}" "oc apply -f ~/models/erebus/v2/k8s/pvc.yaml"

# Optional: create HF token secret for pushing model to HuggingFace
# Uncomment and set your token:
# ssh "${BASTION}" "oc create secret generic erebus-v2-secrets \
#     --from-literal=hf-token=hf_YOUR_TOKEN_HERE \
#     -n machine-learning 2>/dev/null || true"

echo "==> Deleting any previous job..."
ssh "${BASTION}" "oc delete job erebus-v2-train -n machine-learning 2>/dev/null || true"

echo "==> Launching training job..."
ssh "${BASTION}" "oc apply -f ~/models/erebus/v2/k8s/training-job.yaml"

echo "==> Waiting for pod to start..."
sleep 5
ssh "${BASTION}" "oc get pods -l app=erebus-v2 -n machine-learning"

echo ""
echo "========================================="
echo "  Training job submitted!"
echo "========================================="
echo ""
echo "Monitor training:"
echo "  ssh ${BASTION} 'oc logs -f job/erebus-v2-train -n machine-learning'"
echo ""
echo "Check status:"
echo "  ssh ${BASTION} 'oc get pods -l app=erebus-v2 -n machine-learning'"
echo ""
echo "Cleanup when done:"
echo "  ssh ${BASTION} 'oc delete job erebus-v2-train -n machine-learning'"
echo "  ssh ${BASTION} 'oc delete pvc erebus-v2-data -n machine-learning'"
