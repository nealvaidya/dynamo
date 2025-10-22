#!/bin/bash
# Quick deployment script for TensorRT-LLM KVBM bug reproduction on Kubernetes
#
# This script automates the deployment steps outlined in README.md

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required environment variables
if [ -z "$NAMESPACE" ]; then
    print_error "NAMESPACE environment variable is not set"
    echo "Example: export NAMESPACE=trtllm-kvbm-test"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    print_error "HF_TOKEN environment variable is not set"
    echo "Example: export HF_TOKEN=hf_your_token_here"
    exit 1
fi

if [ -z "$RELEASE_VERSION" ]; then
    print_warn "RELEASE_VERSION not set, using default: 0.5.1"
    RELEASE_VERSION="0.5.1"
fi

# Confirm deployment
print_info "=== TensorRT-LLM KVBM Deployment Configuration ==="
echo "Namespace:        $NAMESPACE"
echo "Release Version:  $RELEASE_VERSION"
echo "HF Token:         ${HF_TOKEN:0:10}..."
echo ""
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Deployment cancelled"
    exit 0
fi

# Step 1: Create namespace
print_info "Step 1: Creating namespace ${NAMESPACE}..."
kubectl create namespace ${NAMESPACE} || print_warn "Namespace ${NAMESPACE} already exists"

# Step 2: Create HuggingFace secret
print_info "Step 2: Creating HuggingFace secret..."
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

# Step 3: Create PVC
print_info "Step 3: Creating models PVC..."
cat <<EOF | kubectl apply -n ${NAMESPACE} -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
EOF

# Step 4: Install CRDs
print_info "Step 4: Installing Dynamo CRDs..."
if ! kubectl get crd dynamographdeployments.nvidia.com &> /dev/null; then
    print_info "Downloading and installing CRDs..."
    helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
    helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default
else
    print_warn "CRDs already installed, skipping"
fi

# Step 5: Install Dynamo Platform
print_info "Step 5: Installing Dynamo Platform (namespace-scoped)..."
if ! helm list -n ${NAMESPACE} | grep -q dynamo-platform; then
    print_info "Downloading and installing platform chart..."
    helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
    helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
      --namespace ${NAMESPACE} \
      --set "dynamo-operator.namespaceRestriction.enabled=true" \
      --set "dynamo-operator.namespaceRestriction.targetNamespace=${NAMESPACE}" \
      --set "etcd.image.repository=bitnamilegacy/etcd" \
      --set "etcd.global.security.allowInsecureImages=true"
else
    print_warn "Platform already installed, skipping"
fi

# Wait for platform components
print_info "Waiting for platform components to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=dynamo-operator -n ${NAMESPACE} --timeout=300s || true
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=etcd -n ${NAMESPACE} --timeout=300s || true
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=nats -n ${NAMESPACE} --timeout=300s || true

# Step 6: Update service endpoints in YAML
print_info "Step 6: Updating service endpoints in trtllm_kvbm.yaml..."
cp trtllm_kvbm.yaml trtllm_kvbm.yaml.bak
sed "s/trtllm-kvbm-test/${NAMESPACE}/g" trtllm_kvbm.yaml.bak > trtllm_kvbm.yaml

# Step 7: Deploy the graph
print_info "Step 7: Deploying TensorRT-LLM KVBM graph..."
kubectl apply -f trtllm_kvbm.yaml -n ${NAMESPACE}

# Restore original file
mv trtllm_kvbm.yaml.bak trtllm_kvbm.yaml

print_info "=== Deployment Complete ==="
echo ""
print_info "Next steps:"
echo "1. Monitor deployment status:"
echo "   kubectl get dynamographdeployment -n ${NAMESPACE} -w"
echo ""
echo "2. Watch pods being created:"
echo "   kubectl get pods -n ${NAMESPACE} -w"
echo ""
echo "3. Check worker logs:"
echo "   kubectl logs -f -l dynamo.nvidia.com/component=TRTLLMWorker -n ${NAMESPACE}"
echo ""
echo "4. Check frontend logs:"
echo "   kubectl logs -f -l dynamo.nvidia.com/component=Frontend -n ${NAMESPACE}"

