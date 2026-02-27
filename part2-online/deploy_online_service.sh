#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="ml-deployment-488023"
SERVICE_BASE_NAME="oip-server"
HOST_SUFFIX="$(hostname -s | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
REGION="europe-west4"
REPO="ml-deployment"
IMAGE="europe-west4-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_BASE_NAME}:${HOST_SUFFIX}"

MIN_INSTANCES=1
MAX_INSTANCES=5
MEMORY="2048Mi"
CPU="1"
CONCURRENCY=1

SERVICE_NAME="${SERVICE_BASE_NAME}-${HOST_SUFFIX}"

echo "Authenticating with service account key from GOOGLE_APPLICATION_CREDENTIALS..."
gcloud auth activate-service-account --key-file "${GOOGLE_APPLICATION_CREDENTIALS}"
gcloud --project "${PROJECT_ID}" auth list

echo "Authenticating Docker to Artifact Registry..."
gcloud --project "${PROJECT_ID}" auth configure-docker "europe-west4-docker.pkg.dev" --quiet

echo "Building Docker image '${IMAGE}' locally..."
docker build -t "${IMAGE}" .

echo "Pushing image to Artifact Registry..."
docker push "${IMAGE}"

echo "Deploying service '${SERVICE_NAME}' to Cloud Run..."
gcloud --project "${PROJECT_ID}" run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --concurrency "${CONCURRENCY}" \
  --no-allow-unauthenticated

echo "Deployment complete!"
echo "Service URL:"
gcloud --project "${PROJECT_ID}" run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format="value(status.url)"
