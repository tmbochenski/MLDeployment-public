#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="ml-deployment-488023"
REGION="europe-west4"

TEMP_LOCATION="gs://dataflow-ml-deployment-488023/temp"
STAGING_LOCATION="gs://dataflow-ml-deployment-488023/staging"

JOB_BASE_NAME="sentiment-analysis-pipeline"
JOB_SUFFIX="$(hostname -s | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"

TS="$(date -u +%Y%m%d-%H%M%S)"
JOB_NAME="${JOB_BASE_NAME}-${JOB_SUFFIX}-${TS}"

REPO="ml-deployment"
IMAGE_NAME="beam-batch-worker"
TAG="${TS}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "Authenticating Docker to Artifact Registry..."
gcloud --project "${PROJECT_ID}" auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "Building Dataflow SDK container image '${IMAGE}'..."
docker build -f Dockerfile -t "${IMAGE}" .

echo "Pushing image to Artifact Registry..."
docker push "${IMAGE}"

echo "Submitting Dataflow job '${JOB_NAME}' using SDK container image..."
python batch_inference_pipeline.py \
  --runner DataflowRunner \
  --project "${PROJECT_ID}" \
  --temp_location "${TEMP_LOCATION}" \
  --staging_location "${STAGING_LOCATION}" \
  --region "${REGION}" \
  --job_name "${JOB_NAME}" \
  --sdk_container_image "${IMAGE}" \
  --experiments use_runner_v2 \
  --worker_machine_type=c2d-standard-2 \
  --autoscaling_algorithm=NONE \
  --num_workers=1 \
  --no-wait
