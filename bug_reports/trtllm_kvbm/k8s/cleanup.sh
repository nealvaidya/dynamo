#!/bin/bash
# Cleanup script for TensorRT-LLM KVBM bug reproduction on Kubernetes
#
# This script removes all resources created by the deployment

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

# Parse command line arguments
DELETE_NAMESPACE=false
DELETE_CRDS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --delete-namespace)
            DELETE_NAMESPACE=true
            shift
            ;;
        --delete-crds)
            DELETE_CRDS=true
            shift
            ;;
        --all)
            DELETE_NAMESPACE=true
            DELETE_CRDS=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 [--delete-namespace] [--delete-crds] [--all]"
            echo ""
            echo "Options:"
            echo "  --delete-namespace  Also delete the namespace ${NAMESPACE}"
            echo "  --delete-crds       Also delete Dynamo CRDs (affects entire cluster)"
            echo "  --all               Delete everything including namespace and CRDs"
            exit 1
            ;;
    esac
done

# Confirm cleanup
print_warn "=== TensorRT-LLM KVBM Cleanup Configuration ==="
echo "Namespace:         $NAMESPACE"
echo "Delete namespace:  $DELETE_NAMESPACE"
echo "Delete CRDs:       $DELETE_CRDS"
echo ""
print_warn "This will remove the following resources:"
echo "  - DynamoGraphDeployment: trtllm-deepseek-v3-b200-kvbm"
echo "  - ConfigMap: trtllm-deepseek-v3-kvbm-config"
echo "  - Helm release: dynamo-platform (in ${NAMESPACE})"
if [ "$DELETE_NAMESPACE" = true ]; then
    echo "  - Entire namespace: ${NAMESPACE}"
fi
if [ "$DELETE_CRDS" = true ]; then
    echo "  - Dynamo CRDs (cluster-wide)"
fi
echo ""
read -p "Continue with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Cleanup cancelled"
    exit 0
fi

# Step 1: Delete the DynamoGraphDeployment
print_info "Step 1: Deleting DynamoGraphDeployment..."
kubectl delete -f trtllm_kvbm.yaml -n ${NAMESPACE} --ignore-not-found=true || print_warn "Failed to delete using YAML, trying direct delete..."
kubectl delete dynamographdeployment trtllm-deepseek-v3-b200-kvbm -n ${NAMESPACE} --ignore-not-found=true || print_warn "DynamoGraphDeployment not found"

# Wait a bit for resources to be cleaned up
print_info "Waiting for resources to be deleted..."
sleep 5

# Step 2: Uninstall platform
print_info "Step 2: Uninstalling Dynamo Platform..."
helm uninstall dynamo-platform -n ${NAMESPACE} || print_warn "Platform not installed or already uninstalled"

# Wait for helm resources to be cleaned up
print_info "Waiting for platform resources to be deleted..."
sleep 5

# Step 3: Delete CRDs if requested
if [ "$DELETE_CRDS" = true ]; then
    print_info "Step 3: Uninstalling Dynamo CRDs..."
    helm uninstall dynamo-crds -n default || print_warn "CRDs not installed or already uninstalled"
else
    print_info "Step 3: Skipping CRD deletion (use --delete-crds to remove)"
fi

# Step 4: Delete namespace if requested
if [ "$DELETE_NAMESPACE" = true ]; then
    print_info "Step 4: Deleting namespace ${NAMESPACE}..."
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    print_info "Waiting for namespace deletion to complete..."
    # This can take a while
    kubectl wait --for=delete namespace/${NAMESPACE} --timeout=300s || print_warn "Namespace deletion timed out or already deleted"
else
    print_info "Step 4: Skipping namespace deletion (use --delete-namespace to remove)"
    print_warn "The following resources remain in namespace ${NAMESPACE}:"
    kubectl get all -n ${NAMESPACE} 2>/dev/null || echo "  (namespace may be empty or already deleted)"
fi

print_info "=== Cleanup Complete ==="
echo ""

if [ "$DELETE_NAMESPACE" = false ]; then
    print_info "To delete the namespace manually:"
    echo "  kubectl delete namespace ${NAMESPACE}"
    echo ""
fi

if [ "$DELETE_CRDS" = false ]; then
    print_info "To delete CRDs manually (affects entire cluster):"
    echo "  helm uninstall dynamo-crds -n default"
    echo "  # Or:"
    echo "  kubectl delete crd dynamographdeployments.nvidia.com"
    echo "  kubectl delete crd dynamocomponentdeployments.nvidia.com"
fi

