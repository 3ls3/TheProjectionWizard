#!/usr/bin/env bash
set -eu

IMAGE_TAG=wizard-dev-test
CONTAINER_NAME=pw-local-test

# build (re-use Task 1 script so we're always up to date)
bash scripts/local_docker_test.sh

# kill any previous test container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# run detached and map port (use 8502 to avoid conflicts)
docker run -d --name $CONTAINER_NAME -p 8502:8501 $IMAGE_TAG

# simple health check: wait max 30 s for 200/302
echo -n "⏳ waiting for Streamlit… "
for i in {1..30}; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8502 || true)
  if [[ "$HTTP_CODE" =~ 200|302 ]]; then
    echo "✅ app is up ($HTTP_CODE)."
    docker rm -f $CONTAINER_NAME    # tidy up
    exit 0
  fi
  sleep 1
done

echo "❌ Streamlit never became reachable."
docker logs $CONTAINER_NAME || true
docker rm -f $CONTAINER_NAME
exit 1 