#!/bin/bash
set -euo pipefail

# ── GCP Project Bootstrap Script ───────────────────────────────────────────
# Idempotent setup for projection-wizard GCP deployment
# Requires: gcloud CLI, PROJECT_ID and REGION env vars

echo "🚀 GCP Project Bootstrap - Projection Wizard"
echo "=============================================="

# ── verify gcloud CLI ──────────────────────────────────────────────────────
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Please install Google Cloud SDK first."
    echo "   Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ gcloud CLI found: $(gcloud --version | head -n1)"

# ── check authentication ───────────────────────────────────────────────────
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
    echo "❌ Error: No active gcloud authentication found."
    echo "   Run: gcloud auth login"
    exit 1
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1)
echo "✅ Authenticated as: $ACTIVE_ACCOUNT"

# ── check required environment variables ────────────────────────────────────
if [[ -z "${PROJECT_ID:-}" ]]; then
    echo "❌ Error: PROJECT_ID environment variable is required."
    echo "   Example: export PROJECT_ID=w23-pwiz-v1"
    echo "   Or create a .env file (see .env.example)"
    exit 1
fi

if [[ -z "${REGION:-}" ]]; then
    echo "❌ Error: REGION environment variable is required."
    echo "   Example: export REGION=europe-west1"
    echo "   Or create a .env file (see .env.example)"
    exit 1
fi

echo "✅ PROJECT_ID: $PROJECT_ID"
echo "✅ REGION: $REGION"

# ── set gcloud configuration ───────────────────────────────────────────────
echo ""
echo "📝 Setting gcloud configuration..."
gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"
echo "✅ gcloud config updated"

# ── enable required APIs ───────────────────────────────────────────────────
echo ""
echo "🔧 Enabling required Google Cloud APIs..."

REQUIRED_APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com" 
    "cloudbuild.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    echo "  → Checking $api..."
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        echo "    ✅ Already enabled"
    else
        echo "    🔄 Enabling..."
        gcloud services enable "$api"
        echo "    ✅ Enabled"
    fi
done

# ── create artifact registry repository ─────────────────────────────────────
echo ""
echo "📦 Setting up Artifact Registry..."

REPO_NAME="wizard-repo"
echo "  → Checking repository: $REPO_NAME"

if gcloud artifacts repositories describe "$REPO_NAME" \
   --location="$REGION" \
   --format="value(name)" &>/dev/null; then
    echo "    ✅ Repository already exists"
else
    echo "    🔄 Creating repository..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker images for Projection Wizard"
    echo "    ✅ Repository created"
fi

# ── summary and next steps ──────────────────────────────────────────────────
echo ""
echo "🎉 GCP Project Bootstrap Complete!"
echo "=================================="
echo ""
echo "📋 Configuration Summary:"
echo "  • Project: $PROJECT_ID"
echo "  • Region: $REGION"
echo "  • Artifact Registry: $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
echo ""
echo "🚀 Next Steps:"
echo "  1. Build your Docker image:"
echo "     docker build -t projection-wizard ."
echo ""
echo "  2. Tag and push to registry:"
echo "     docker tag projection-wizard $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/projection-wizard:latest"
echo "     docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/projection-wizard:latest"
echo ""
echo "  3. Deploy to Cloud Run:"
echo "     gcloud run deploy projection-wizard \\"
echo "       --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/projection-wizard:latest \\"
echo "       --region $REGION \\"
echo "       --allow-unauthenticated"
echo ""
echo "✅ All systems ready for deployment!" 