#!/bin/bash
set -e

# Usage: ./build-local-images.sh [scheduler|fetcher|optimizer|executor ...]
# If no arguments are provided, all images are built.

IMAGES=(scheduler fetcher optimizer executor)
DOCKERFILES=(
  [scheduler]=dockerfiles/Dockerfile.scheduler
  [fetcher]=dockerfiles/Dockerfile.fetcher
  [optimizer]=dockerfiles/Dockerfile.optimizer
  [executor]=dockerfiles/Dockerfile.executor
)
TAGS=(
  [scheduler]=svenhockers/portfoliobalancer:scheduler
  [fetcher]=svenhockers/portfoliobalancer:fetcher
  [optimizer]=svenhockers/portfoliobalancer:optimizer
  [executor]=svenhockers/portfoliobalancer:executor
)

# If no args, build all
if [ $# -eq 0 ]; then
  BUILD_LIST=("${IMAGES[@]}")
else
  BUILD_LIST=("$@")
fi

for IMAGE in "${BUILD_LIST[@]}"; do
  DOCKERFILE=${DOCKERFILES[$IMAGE]}
  TAG=${TAGS[$IMAGE]}
  if [ -z "$DOCKERFILE" ] || [ -z "$TAG" ]; then
    echo "Unknown image: $IMAGE. Valid options: scheduler, fetcher, optimizer, executor"
    exit 1
  fi
  echo "Building $IMAGE image..."
  docker build -f "$DOCKERFILE" -t "$TAG" .
done

echo "All set for local testing, images build and ready to be deployed" 
echo "May the odds ever be in your favor!!"