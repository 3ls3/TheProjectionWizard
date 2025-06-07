#!/bin/bash

# ── Projection Wizard — Push Image to Artifact Registry ───────────────────
set -eu

# ── Load environment variables ─────────────────────────────────────────────
# Store any existing env vars before sourcing .env
_PROJECT_ID=${PROJECT_ID:-}
_REGION=${REGION:-}
_IMAGE_TAG=${IMAGE_TAG:-}

if [[ -f .env ]]; then
    source .env
fi

# Use CLI args over .env file
PROJECT_ID=${_PROJECT_ID:-${PROJECT_ID:-}}
REGION=${_REGION:-${REGION:-}}
IMAGE_TAG=${_IMAGE_TAG:-${IMAGE_TAG:-wizard:latest}}
REPO_NAME="wizard-repo"

# ── Validation ─────────────────────────────────────────────────────────────
if [[ -z "$PROJECT_ID" ]]; then
    echo "❌ ERROR: PROJECT_ID not set. Add to .env or export PROJECT_ID=your-project-id"
    exit 1
fi

if [[ -z "$REGION" ]]; then
    echo "❌ ERROR: REGION not set. Add to .env or export REGION=your-region"
    exit 1
fi

echo "🔧 Using PROJECT_ID=$PROJECT_ID, REGION=$REGION, IMAGE_TAG=$IMAGE_TAG"

# ── Check Docker auth ──────────────────────────────────────────────────────
REGISTRY_HOST="${REGION}-docker.pkg.dev"
if ! docker login --username=oauth2accesstoken --password="$(gcloud auth print-access-token)" "$REGISTRY_HOST" 2>/dev/null; then
    echo "❌ ERROR: Docker auth failed. Run 'gcloud auth configure-docker ${REGISTRY_HOST}' first"
    exit 1
fi

# ── Build image if not present ─────────────────────────────────────────────
if ! docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^$IMAGE_TAG$"; then
    echo "🔨 Building image $IMAGE_TAG for Cloud Run (linux/amd64)..."
    docker build --platform linux/amd64 -t "$IMAGE_TAG" .
fi

# ── Tag and push ───────────────────────────────────────────────────────────
TARGET="${REGISTRY_HOST}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_TAG}"

echo "🏷️  Tagging $IMAGE_TAG -> $TARGET"
docker tag "$IMAGE_TAG" "$TARGET"

echo "🚀 Pushing to Artifact Registry..."
docker push "$TARGET"

echo "✅ Pushed to $TARGET" 