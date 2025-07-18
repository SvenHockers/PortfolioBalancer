#!/bin/bash
set -e

IMAGES=(scheduler fetcher optimizer executor)
DOCKERFILES=(dockerfiles/Dockerfile.scheduler dockerfiles/Dockerfile.fetcher dockerfiles/Dockerfile.optimizer dockerfiles/Dockerfile.executor)
TAGS=(svenhockers/portfoliobalancer:scheduler svenhockers/portfoliobalancer:fetcher svenhockers/portfoliobalancer:optimizer svenhockers/portfoliobalancer:executor)

get_dockerfile() {
    local image=$1
    for i in "${!IMAGES[@]}"; do
        if [[ "${IMAGES[$i]}" == "$image" ]]; then
            echo "${DOCKERFILES[$i]}"
            return 0
        fi
    done
    return 1
}

get_tag() {
    local image=$1
    for i in "${!IMAGES[@]}"; do
        if [[ "${IMAGES[$i]}" == "$image" ]]; then
            echo "${TAGS[$i]}"
            return 0
        fi
    done
    return 1
}

if [ $# -eq 0 ]; then
  BUILD_LIST=("${IMAGES[@]}")
else
  BUILD_LIST=("$@")
fi

for IMAGE in "${BUILD_LIST[@]}"; do
  DOCKERFILE=$(get_dockerfile "$IMAGE")
  TAG=$(get_tag "$IMAGE")
  if [ -z "$DOCKERFILE" ] || [ -z "$TAG" ]; then
    echo "Unknown image: $IMAGE. Valid options: scheduler, fetcher, optimizer, executor"
    exit 1
  fi
  echo "Building $IMAGE image..."
  docker build -f "$DOCKERFILE" -t "$TAG" .
done

echo "All set for local testing, images build and ready to be deployed" 
echo "May the odds ever be in your favor!!"
