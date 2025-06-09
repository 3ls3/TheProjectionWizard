#!/usr/bin/env bash
set -e

IMAGE_TAG=wizard-dev-test

echo "⏳  Building image…"
docker build -t $IMAGE_TAG .

echo "✅  Build succeeded." 