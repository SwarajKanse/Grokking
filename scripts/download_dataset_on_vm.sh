#!/usr/bin/env bash
set -e

ACCOUNT_NAME="${AZURE_STORAGE_ACCOUNT:-stber2026india}"
ACCOUNT_KEY="${1:-$AZURE_STORAGE_KEY}"
CONTAINER="${AZURE_STORAGE_CONTAINER:-ber-dataset}"
DEST_DIR="$HOME/Grokking/dataset"

mkdir -p "$DEST_DIR"
echo "=== Downloading dataset from Azure Blob to VM ==="
az storage blob download-batch \
  --account-name "$ACCOUNT_NAME" \
  --account-key "$ACCOUNT_KEY" \
  --source "$CONTAINER" \
  --destination "$DEST_DIR"

echo "=== Verifying downloaded files ==="
ls -lh "$DEST_DIR"
ls -lh "$DEST_DIR/train" || true
ls -lh "$DEST_DIR/test" || true
echo "=== Dataset download complete ==="
