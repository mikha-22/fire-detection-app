#!/bin/bash

# This script creates a self-contained model archive (model.mar)
# that includes the model, the handler, and the requirements,
# following the official Google Vertex AI pre-built container pattern.

echo "Step 1: Generating model.pth..."
python src/ml_model/fire_detection_model.py
if [ ! -f "src/ml_model/model.pth" ]; then
    echo "ERROR: src/ml_model/model.pth not found!"
    exit 1
fi
echo "model.pth generated successfully."

echo "Step 2: Archiving model into model.mar..."
TEMP_EXPORT_PATH="src/ml_model/model_store_temp"
rm -rf "$TEMP_EXPORT_PATH"
mkdir -p "$TEMP_EXPORT_PATH"

# Create the self-contained MAR file.
# We provide all the necessary files directly to the archiver.
torch-model-archiver --model-name model \
                     --version 1.0 \
                     --model-file src/ml_model/fire_detection_model.py \
                     --serialized-file src/ml_model/model.pth \
                     --handler src/ml_model/handler.py \
                     --requirements-file src/ml_model/requirements.txt \
                     --export-path "$TEMP_EXPORT_PATH" \
                     --force

if [ ! -f "$TEMP_EXPORT_PATH/model.mar" ]; then
    echo "ERROR: $TEMP_EXPORT_PATH/model.mar not found!"
    rm -rf "$TEMP_EXPORT_PATH"
    exit 1
fi
echo "$TEMP_EXPORT_PATH/model.mar created successfully."

echo "Step 3: Moving model.mar to its final location..."
mv "$TEMP_EXPORT_PATH/model.mar" "src/ml_model/model.mar"
rm -rf "$TEMP_EXPORT_PATH"
echo "model.mar moved to src/ml_model/model.mar."

echo "Model archiving process completed successfully."
