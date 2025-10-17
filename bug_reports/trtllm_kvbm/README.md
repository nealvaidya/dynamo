# TensorRT-LLM with KVBM Docker Compose Setup

This directory contains a Docker Compose configuration converted from the Kubernetes DynamoGraphDeployment for local testing.

## Files

- `docker-compose.yaml` - Main Docker Compose configuration
- `kvbm_config.yaml` - KVBM configuration file (mounted into containers)
- `build-image.sh` - Convenience script to build ARM64 images
- `trtllm_kvbm.yaml` - Original Kubernetes configuration
- `MPI_FIX.md` - Detailed explanation of MPI configuration for multi-GPU

## Prerequisites

1. Docker with GPU support (nvidia-docker2 or Docker with NVIDIA Container Toolkit)
2. NVIDIA GPU with at least 4 GPUs available per worker
3. HuggingFace account and token for model access

## Building the Docker Images

Since you're running on ARM architecture, you'll need to build the images locally as the published images are x86-64 only.

### Quick Build (Recommended)

Use the convenience script from this directory:

```bash
./build-image.sh
```

This will build the `dynamo:latest-trtllm` image for ARM64.

### Manual Build

Alternatively, build from the repository root:

```bash
cd ../../container

# Build TensorRT-LLM image for ARM64
./build.sh --framework trtllm --platform linux/arm64

# This creates the image: dynamo:latest-trtllm
```

**Note:** The build process may take 30-60 minutes depending on your system. It will:
- Download base images
- Compile TensorRT-LLM for ARM64
- Install all Dynamo dependencies
- Create an optimized runtime image

**Build Options:**
- `--no-cache` - Force rebuild without using cached layers
- `--tag custom-tag` - Use a custom image tag instead of `latest-trtllm`
- `--dry-run` - Show the Docker commands without executing them

If you use a custom tag, update the `image:` fields in `docker-compose.yaml` accordingly.

## Setup

1. **Build the Docker image** (see "Building the Docker Images" section above)

2. Set your HuggingFace token as an environment variable:
   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

   Or create a `.env` file in this directory:
   ```bash
   HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

3. Ensure you have sufficient resources:
   - 4 GPUs (for 1 worker)
   - 500GB shared memory per worker
   - 8GB CPU memory for KV cache offloading
   - 8GB disk space for KV cache offloading

## Running

Start all services:
```bash
docker-compose up -d
```

View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f worker-1
docker-compose logs -f frontend
```

Check service status:
```bash
docker-compose ps
```

Test the API:
```bash
curl http://localhost:8000/v1/models
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
- Command: `python3 -m dynamo.frontend --http-port 8000`
- Router mode: KV
- Port: 8000 (OpenAI-compatible HTTP API)
- Depends on: nats-server, etcd-server, worker-1
- Environment:
  - `NATS_SERVER=nats://nats-server:4222`
  - `ETCD_ENDPOINTS=http://etcd-server:2379`
  - `DYN_LOG=debug` - Enable debug-level logging
  - `DYN_LOGGING_JSONL=false` - Use readable log format

### Worker-1
- Container: `trtllm-worker-1`
- Image: `dynamo:latest-trtllm` (locally built)
- Model: `nvidia/DeepSeek-V3-0324-FP4`
- Depends on: nats-server, etcd-server
- Resources:
  - 4 GPUs
  - 500GB shared memory
  - Tensor parallel size: 4
  - Expert parallel size: 4
  - Max batch size: 128
- Environment:
  - `NATS_SERVER`, `ETCD_ENDPOINTS` - Service discovery
  - `DYN_LOG=debug` - Enable debug-level logging
  - `DYN_LOGGING_JSONL=false` - Use readable log format
  - `NCCL_IB_DISABLE=1` - Disable InfiniBand
  - `NCCL_P2P_LEVEL=NVL` - Use NVLink for GPU-to-GPU
- Special configurations for multi-GPU MPI:
  - `ipc: host` - Required for inter-process communication
  - `ulimits` - Memory lock, stack size, and file descriptors
  - `cap_add: SYS_PTRACE` - Required for debugging and MPI

## KVBM Features Enabled

- CPU cache offloading: 8GB
- Disk cache offloading: 8GB
- Metrics enabled (port 6880)
- Leader-worker initialization timeout: 1200 seconds

## API Access

Once the services are running, the frontend exposes an OpenAI-compatible API on port 8000:

```bash
# List available models
curl http://localhost:8000/v1/models

# Chat completion request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/DeepSeek-V3-0324-FP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Troubleshooting

### Worker hangs at "start MpiSession"

If the worker gets stuck at:
```
[TRT-LLM] [I] start MpiSession with 4 workers
```

This is an MPI initialization hang. The docker-compose now includes required MPI configurations:
- `ipc: host` - Critical for MPI inter-process communication
- `ulimits` - Proper memory and file descriptor limits
- `cap_add: SYS_PTRACE` - Required capability for MPI
- NCCL environment variables for multi-GPU communication

If still hanging, check:
1. Ensure you're using the updated docker-compose.yaml with these settings
2. Restart containers: `docker-compose down && docker-compose up -d`
3. Check GPU connectivity: `nvidia-smi nvlink --status`

### Other Common Issues

1. **Check etcd/NATS connectivity**: Ensure `NATS_SERVER` and `ETCD_ENDPOINTS` are correctly set
2. **Check logs**: `docker-compose logs worker-1` or `docker-compose logs frontend`
3. **Verify GPU access**: Ensure NVIDIA Container Toolkit is installed and working
4. **Check shared memory**: Ensure your system has enough memory for the 500GB shm_size

### Debug Logging

Debug logging is enabled by default with:
- `DYN_LOG=debug` - Shows detailed debugging information
- `DYN_LOGGING_JSONL=false` - Uses human-readable log format

To change log levels:
```yaml
environment:
  - DYN_LOG=trace  # Most verbose: trace, debug, info, warn, error
  - DYN_LOG=info   # Less verbose (default in production)
```

To enable JSONL structured logging (useful for log aggregation):
```yaml
environment:
  - DYN_LOGGING_JSONL=true
```

For NCCL-specific debugging:
```yaml
environment:
  - NCCL_DEBUG=INFO  # NCCL communication logs
```

## Notes

- **Images**: Using locally built `dynamo:latest-trtllm` (not the published x86-64 images)
- **Models**: The `models-volume` Docker volume will cache downloaded models
- **KVBM Infrastructure**: Requires etcd and nats services (included in this compose file)
  - NATS provides messaging between workers and the frontend
  - etcd provides distributed coordination for KVBM operations
- **Service Discovery**: Workers and frontend discover each other via etcd
- **GPU Requirements**: Adjust GPU count if you have fewer GPUs available (modify `count` in the deploy section)
- **Memory**: Shared memory size is set to 500GB; adjust based on your system's available memory
- **ARM Build Time**: Initial image build can take 30-60 minutes

