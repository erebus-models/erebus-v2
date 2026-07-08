#!/usr/bin/env bash
set -euo pipefail

# Build the training container via OpenShift binary build.
# The internal registry isn't reachable from the bastion directly,
# so we use `oc new-build` / `oc start-build` which builds inside the cluster.
#
# Run from your laptop (inside the erebus-v2 directory):
#   bash scripts/build_and_push.sh

BASTION="soyr-redhat@169.62.21.215"
REMOTE_DIR="~/models/erebus/v2"
IMAGE_STREAM="erebus-v2"
NAMESPACE="machine-learning"

echo "==> Syncing project to bastion..."
rsync -avz --exclude '.git' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/id_ed25519" \
    ./ "${BASTION}:${REMOTE_DIR}/"

echo "==> Setting up OpenShift build..."
ssh -i ~/.ssh/id_ed25519 "${BASTION}" bash -s <<'REMOTE'
set -euo pipefail
cd ~/models/erebus/v2

NAMESPACE="machine-learning"
IMAGE_STREAM="erebus-v2"

# Create ImageStream if it doesn't exist
oc get is ${IMAGE_STREAM} -n ${NAMESPACE} 2>/dev/null || \
    oc create imagestream ${IMAGE_STREAM} -n ${NAMESPACE}

# Create or update the BuildConfig for binary builds
oc get bc ${IMAGE_STREAM} -n ${NAMESPACE} 2>/dev/null || \
    oc new-build --name=${IMAGE_STREAM} \
        --binary \
        --strategy=docker \
        --to=${IMAGE_STREAM}:latest \
        -n ${NAMESPACE}

echo "==> Starting binary build (uploading context + building in cluster)..."
oc start-build ${IMAGE_STREAM} \
    --from-dir=. \
    --follow \
    --wait \
    -n ${NAMESPACE}

echo ""
echo "==> Build complete! Image available at:"
echo "    image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_STREAM}:latest"
REMOTE

echo "==> Done!"
