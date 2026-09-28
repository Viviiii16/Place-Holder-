#!/usr/bin/env bash
# Deploys the OilSpill Intelligence API to Google Cloud Run.
# Run from Google Cloud Shell inside this folder (best_oil_deeplabv3plus.pth must be here).
set -euo pipefail

SERVICE="oilspill-api"
REGION="${REGION:-us-central1}"

[ -f best_oil_deeplabv3plus.pth ] || { echo "❌ best_oil_deeplabv3plus.pth not found in $(pwd)"; exit 1; }
[ -n "${CDS_API_KEY:-}" ] && [ -n "${CMEMS_USERNAME:-}" ] && [ -n "${CMEMS_PASSWORD:-}" ] || {
  echo "❌ Set your credentials first, e.g.:"
  echo "   export CDS_API_KEY='...' CMEMS_USERNAME='...' CMEMS_PASSWORD='...'"
  exit 1; }

echo "▶ Enabling required Google Cloud services (first run only)…"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

echo "▶ Building and deploying (about 10–15 min the first time)…"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 4Gi \
  --no-cpu-throttling \
  --min-instances 0 --max-instances 1 \
  --timeout 300 \
  --set-env-vars "CDS_API_KEY=${CDS_API_KEY},CMEMS_USERNAME=${CMEMS_USERNAME},CMEMS_PASSWORD=${CMEMS_PASSWORD}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "✅ Deployed: $URL"
echo "   Health:   $URL/api/health"
echo
echo "Use this for BOTH website variables:"
echo "   VITE_DETECTION_API_URL=$URL"
echo "   VITE_ATTRIBUTION_API_URL=$URL"
