#!/bin/bash

# ── Projection Wizard — Deploy to Cloud Run ───────────────────────────────
set -eu

# ── Load environment variables ─────────────────────────────────────────────
if [[ -f .env ]]; then
    source .env
fi

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-}
IMAGE_TAG=${IMAGE_TAG:-wizard:latest}
REPO_NAME=${REPO_NAME:-wizard-repo}
SERVICE_NAME="projection-wizard"

# ── Validation ─────────────────────────────────────────────────────────────
if [[ -z "$PROJECT_ID" ]]; then
    echo "❌ ERROR: PROJECT_ID not set. Add to .env or export PROJECT_ID=your-project-id"
    exit 1
fi

if [[ -z "$REGION" ]]; then
    echo "❌ ERROR: REGION not set. Add to .env or export REGION=your-region"
    exit 1
fi

echo "🔧 Deploying to Cloud Run..."
echo "   PROJECT_ID: $PROJECT_ID"
echo "   REGION: $REGION"
echo "   IMAGE_TAG: $IMAGE_TAG"
echo "   SERVICE: $SERVICE_NAME"

# ── Build image URL ────────────────────────────────────────────────────────
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_TAG}"
echo "   IMAGE_URL: $IMAGE_URL"

# ── Deploy to Cloud Run ────────────────────────────────────────────────────
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE_URL" \
  --platform=managed \
  --region="$REGION" \
  --allow-unauthenticated \
  --cpu=1 \
  --memory=2Gi \
  --timeout=900 \
  --max-instances=2

# ── Get public URL ─────────────────────────────────────────────────────────
echo "🌐 Getting service URL..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --platform=managed \
  --region="$REGION" \
  --format="value(status.url)")

echo ""
echo "✅ Deployment successful!"
echo "🌍 Public URL: $SERVICE_URL"
echo ""
echo "📱 You can now access the Projection Wizard at:"
echo "   $SERVICE_URL" 