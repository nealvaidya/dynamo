#!/usr/bin/env bash
# Build TensorRT-LLM ARM image for Docker Compose testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Building TensorRT-LLM image for ARM64..."
echo "This may take 30-60 minutes on the first build."
echo ""

cd "$REPO_ROOT/container"

# Build the image
./build.sh --framework trtllm --platform linux/arm64 "$@"

echo ""
echo "✓ Build complete!"
echo "  Image: dynamo:latest-trtllm"
echo ""
echo "Next steps:"
echo "  1. Set your HuggingFace token: export HUGGING_FACE_HUB_TOKEN=your_token"
echo "  2. Run: cd $SCRIPT_DIR && docker-compose up -d"

