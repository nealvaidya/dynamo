# TensorRT-LLM with KVBM Docker Compose Setup

This directory contains a Docker Compose configuration converted from the Kubernetes DynamoGraphDeployment for local testing.

## Files

- `docker-compose.yaml` - Main Docker Compose configuration
- `kvbm_config.yaml` - KVBM configuration file (mounted into containers)
- `build-image.sh` - Convenience script to build ARM64 images
- `trtllm_kvbm.yaml` - Original Kubernetes configuration

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
- Depends on: nats-server, etcd-server, worker-1

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

## KVBM Features Enabled

- CPU cache offloading: 8GB
- Disk cache offloading: 8GB
- Metrics enabled (port 6880)
- Leader-worker initialization timeout: 1200 seconds

## Notes

- **Images**: Using locally built `dynamo:latest-trtllm` (not the published x86-64 images)
- **Models**: The `models-volume` Docker volume will cache downloaded models
- **KVBM Infrastructure**: Requires etcd and nats services (included in this compose file)
  - NATS provides messaging between workers and the frontend
  - etcd provides distributed coordination for KVBM operations
- **GPU Requirements**: Adjust GPU count if you have fewer GPUs available (modify `count` in the deploy section)
- **Memory**: Shared memory size is set to 500GB; adjust based on your system's available memory
- **ARM Build Time**: Initial image build can take 30-60 minutes

