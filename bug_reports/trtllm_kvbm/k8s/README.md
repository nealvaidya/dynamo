# TensorRT-LLM with KVBM - Kubernetes Reproduction

This directory contains instructions and configuration to reproduce the TensorRT-LLM KVBM bug on Kubernetes.

## Overview

This setup deploys a TensorRT-LLM inference service with KV Block Management (KVBM) enabled, along with a namespace-scoped Dynamo operator and required infrastructure (etcd, NATS).

## Prerequisites

1. **Kubernetes Cluster**
   - Kubernetes v1.24+
   - At least 2 nodes with 4 B200 GPUs each (or equivalent)
   - GPU operator installed (NVIDIA drivers available)

2. **Required Tools**
   ```bash
   kubectl version --client  # v1.24+
   helm version             # v3.0+
   ```

3. **Persistent Volume Claims**
   - A PVC named `vllm-models-pvc` must exist for model storage
   - See [Creating the Models PVC](#creating-the-models-pvc) below

4. **HuggingFace Token**
   - Required for accessing the DeepSeek-V3 model
   - Must have access to `nvidia/DeepSeek-V3-0324-FP4`

5. **Set Environment Variables**
   ```bash
   export NAMESPACE=trtllm-kvbm-test
   export RELEASE_VERSION=0.5.1
   export HF_TOKEN=your_huggingface_token_here
   ```

## Setup Instructions

### Step 1: Create Namespace

```bash
kubectl create namespace ${NAMESPACE}
```

### Step 2: Create HuggingFace Secret

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

### Step 3: Creating the Models PVC

Create a PVC for model storage (adjust size and storage class as needed):

```bash
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
  # storageClassName: your-storage-class  # Uncomment and set if needed
EOF
```

**Note:** For multi-node access, use a storage class that supports `ReadWriteMany` (e.g., NFS, CephFS, or cloud provider-specific solutions).

### Step 4: Install Dynamo CRDs

Install the Custom Resource Definitions required by the operator:

```bash
# Download CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz

# Install CRDs (these are cluster-wide)
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# Verify CRDs installation
kubectl get crd | grep dynamo
```

Expected output:
```
dynamocomponentdeployments.nvidia.com
dynamographdeployments.nvidia.com
```

### Step 5: Install Dynamo Platform (Namespace-Scoped Operator)

Install the Dynamo platform with the operator restricted to only watch the `${NAMESPACE}` namespace:

```bash
# Download platform chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz

# Install with namespace restriction
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.namespaceRestriction.enabled=true" \
  --set "dynamo-operator.namespaceRestriction.targetNamespace=${NAMESPACE}" \
  --set "etcd.image.repository=bitnamilegacy/etcd" \
  --set "etcd.global.security.allowInsecureImages=true"
```

**What this does:**
- Deploys the Dynamo operator scoped to only watch the `${NAMESPACE}` namespace
- Deploys etcd and NATS services required by KVBM
- All components are isolated to the test namespace

**Verify installation:**
```bash
kubectl get pods -n ${NAMESPACE}

# Expected pods:
# - dynamo-kubernetes-operator-* (Running)
# - dynamo-platform-etcd-* (Running)
# - dynamo-platform-nats-* (Running)
```

Wait for all pods to be in `Running` state:
```bash
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=dynamo-operator -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=etcd -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=nats -n ${NAMESPACE} --timeout=300s
```

### Step 6: Update Service Endpoints (If Using Different Namespace)

If you're using a namespace other than `trtllm-kvbm-test`, you need to update the NATS and etcd endpoints in `trtllm_kvbm.yaml`:

**Option A: Using sed (Linux/Mac)**
```bash
sed -i.bak "s/trtllm-kvbm-test/${NAMESPACE}/g" trtllm_kvbm.yaml
```

**Option B: Manual Edit**
Edit `trtllm_kvbm.yaml` and replace both occurrences of `trtllm-kvbm-test` with your namespace:
- Line ~36: `NATS_SERVER` value
- Line ~38: `ETCD_ENDPOINTS` value

### Step 7: Deploy the KVBM Configuration and Graph

Apply the TensorRT-LLM KVBM deployment configuration:

```bash
kubectl apply -f trtllm_kvbm.yaml -n ${NAMESPACE}
```

This creates:
1. **ConfigMap**: `trtllm-deepseek-v3-kvbm-config` - Contains KVBM configuration
2. **DynamoGraphDeployment**: `trtllm-deepseek-v3-b200-kvbm` - Defines the inference graph
   - 1 Frontend replica
   - 2 TRTLLMWorker replicas (4 GPUs each)

**Monitor deployment:**
```bash
# Watch DynamoGraphDeployment status
kubectl get dynamographdeployment -n ${NAMESPACE} -w

# Watch pods being created
kubectl get pods -n ${NAMESPACE} -w
```

### Step 8: Monitor Worker Logs

Watch the worker logs to observe the bug:

```bash
# List all pods
kubectl get pods -n ${NAMESPACE}

# Watch logs for worker-0
kubectl logs -f <trtllm-worker-pod-name> -n ${NAMESPACE}

# Or follow logs for all workers
kubectl logs -f -l dynamo.nvidia.com/component=TRTLLMWorker -n ${NAMESPACE}
```

## Expected Behavior vs. Bug

### Expected Behavior
Workers should:
1. Initialize MPI session successfully
2. Load the model
3. Start accepting requests
4. KVBM should enable CPU and disk offloading

### Bug Symptoms
Look for symptoms such as:
- Workers hanging during initialization
- MPI session startup failures
- KVBM initialization issues
- Timeout errors in leader-worker initialization

## Debugging

### Check DynamoGraphDeployment Status

```bash
kubectl describe dynamographdeployment trtllm-deepseek-v3-b200-kvbm -n ${NAMESPACE}
```

### Check Worker Pod Events

```bash
kubectl describe pod -l dynamo.nvidia.com/component=TRTLLMWorker -n ${NAMESPACE}
```

### Check Frontend Logs

```bash
kubectl logs -f -l dynamo.nvidia.com/component=Frontend -n ${NAMESPACE}
```

### Check etcd and NATS

```bash
# Check etcd logs
kubectl logs -f -l app.kubernetes.io/name=etcd -n ${NAMESPACE}

# Check NATS logs
kubectl logs -f -l app.kubernetes.io/name=nats -n ${NAMESPACE}
```

### Get KVBM Metrics

If metrics are enabled (they are by default), check the KVBM metrics endpoint:

```bash
# Port-forward to worker metrics port
kubectl port-forward <worker-pod-name> 6880:6880 -n ${NAMESPACE}

# In another terminal
curl http://localhost:6880/metrics
```

### Interactive Debugging

Get a shell in a worker pod:

```bash
kubectl exec -it <worker-pod-name> -n ${NAMESPACE} -- /bin/bash
```

## Configuration Details

### KVBM Configuration

The KVBM configuration in the ConfigMap includes:
- **Backend**: PyTorch (required for KVBM with TRT-LLM)
- **CUDA Graphs**: Disabled (not supported with KVBM)
- **Partial Reuse**: Disabled (increases cache hits)
- **GPU Memory**: 80% allocated for KV cache
- **CPU Cache**: 128GB offloading enabled
- **Disk Cache**: 8GB offloading enabled
- **Attention DP**: Enabled
- **Chunked Prefill**: Enabled
- **MoE Backend**: TRTLLM
- **Speculative Decoding**: MTP with 3 layers

### Worker Environment Variables

Key environment variables set for workers:
- `DYN_ROUTER_MODE=kv` - Enable KV routing mode
- `DYN_KVBM_BARRIER_ID_PREFIX` - Unique ID per replica (from pod name)
- `DYN_KVBM_CPU_CACHE_GB=128` - 128GB CPU cache
- `DYN_KVBM_DISK_CACHE_GB=8` - 8GB disk cache
- `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS=1200` - 20 minute timeout
- `DYN_KVBM_METRICS=true` - Enable metrics on port 6880
- `TRTLLM_ENABLE_PDL=1` - Enable parallel data loading
- `TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=True` - Disable parallel weight loading

## Cleanup

To remove the deployment:

```bash
# Delete the DynamoGraphDeployment
kubectl delete -f trtllm_kvbm.yaml -n ${NAMESPACE}

# Uninstall the platform
helm uninstall dynamo-platform -n ${NAMESPACE}

# Uninstall CRDs (if no longer needed)
helm uninstall dynamo-crds -n default

# Delete the namespace
kubectl delete namespace ${NAMESPACE}
```

## Node Selector Configuration

The current configuration requires nodes with the label `gpu-type.northflank.com/b200: "true"`. If your cluster uses different labels:

1. Check your GPU node labels:
   ```bash
   kubectl get nodes --show-labels
   ```

2. Update the `nodeSelector` in `trtllm_kvbm.yaml`:
   ```yaml
   extraPodSpec:
     nodeSelector:
       # Replace with your GPU node labels
       your-gpu-label: "true"
   ```

Or remove the `nodeSelector` entirely if not needed.

## Storage Class Configuration

If your cluster doesn't have a default storage class, you need to specify one:

1. List available storage classes:
   ```bash
   kubectl get storageclass
   ```

2. Update the PVC creation in Step 3 to specify the storage class:
   ```yaml
   spec:
     storageClassName: your-storage-class-name
   ```

## Additional Notes

- **Namespace Isolation**: The operator is configured to only watch resources in the `${NAMESPACE}` namespace. This allows for isolated testing without affecting other deployments.
- **Resource Requirements**: Each worker requires 4 GPUs and 500Gi shared memory. Adjust based on your hardware.
- **Timeout Settings**: The KVBM initialization timeout is set to 1200 seconds (20 minutes) to account for model downloading and memory allocation.
- **Model Path**: Uses the HuggingFace model `nvidia/DeepSeek-V3-0324-FP4` (FP4 quantized for efficiency).

