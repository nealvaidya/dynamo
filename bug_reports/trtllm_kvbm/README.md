# TensorRT-LLM with KVBM Docker Compose Setup

This directory contains a Docker Compose configuration converted from the Kubernetes DynamoGraphDeployment for local testing.

## Files

- `docker-compose.yaml` - Main Docker Compose configuration
- `kvbm_config.yaml` - KVBM configuration file (mounted into containers)
- `trtllm_kvbm.yaml` - Original Kubernetes configuration

## Prerequisites

1. Docker with GPU support (nvidia-docker2 or Docker with NVIDIA Container Toolkit)
2. NVIDIA GPU with at least 4 GPUs available per worker (8 GPUs total for 2 workers)
3. HuggingFace account and token for model access

## Setup

1. Set your HuggingFace token as an environment variable:
   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

   Or create a `.env` file in this directory:
   ```bash
   HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

2. Ensure you have sufficient resources:
   - 8 GPUs total (4 per worker)
   - 500GB shared memory per worker
   - 128GB CPU memory for KV cache offloading per worker
   - 8GB disk space for KV cache offloading per worker

## Running

Start all services:
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f
```

Stop all services:
```bash
docker-compose down
```

## Services

### NATS
- Container: `dynamo-nats`
- Image: `nats:2.11.4`
- Purpose: Messaging system for KVBM communication
- Ports: 4222 (client), 6222 (routing), 8222 (monitoring)
- Features: JetStream enabled (`-js`)

### etcd
- Container: `dynamo-etcd`
- Image: `bitnamilegacy/etcd:3.6.1`
- Purpose: Distributed key-value store for KVBM coordination
- Ports: 2379 (client), 2380 (peer)
- Authentication: Disabled for testing (`ALLOW_NONE_AUTHENTICATION=yes`)

### Frontend
- Container: `trtllm-frontend`
- Router mode: KV
- Depends on: nats-server, etcd-server, both workers

### Worker-1 and Worker-2
- Containers: `trtllm-worker-1`, `trtllm-worker-2`
- Model: `nvidia/DeepSeek-V3-0324-FP4`
- Depends on: nats-server, etcd-server
- Each worker uses:
  - 4 GPUs
  - 500GB shared memory
  - Tensor parallel size: 4
  - Expert parallel size: 4
  - Max batch size: 128

## KVBM Features Enabled

- CPU cache offloading: 128GB per worker
- Disk cache offloading: 8GB per worker
- Metrics enabled (port 6880)
- Leader-worker initialization timeout: 1200 seconds

## Notes

- The `models-volume` Docker volume will cache downloaded models
- KVBM requires etcd and nats services (included in this compose file)
- NATS provides messaging between workers and the frontend
- etcd provides distributed coordination for KVBM operations
- Adjust GPU count if you have fewer GPUs available (modify `count` in the deploy section)
- Shared memory size is set to 500GB; adjust based on your system's available memory

