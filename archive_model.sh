#!/bin/bash

# Ensure you are in the project root directory
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

# --- CORRECTED PART ---
# Copy files from their correct location (src/ml_model/) to the root.
cp src/ml_model/handler.py .
cp src/ml_model/fire_detection_model.py .

# Use the correct path for the requirements file: src/ml_model/requirements.txt
torch-model-archiver --model-name model \
                     --version 1.0 \
                     --model-file fire_detection_model.py \
                     --serialized-file src/ml_model/model.pth \
                     --handler handler.py \
                     --requirements-file src/ml_model/requirements.txt \
                     --export-path "$TEMP_EXPORT_PATH" \
                     --force

# Clean up the temporarily copied files
rm handler.py
rm fire_detection_model.py
# --- END CORRECTION ---


if [ ! -f "$TEMP_EXPORT_PATH/model.mar" ]; then
    echo "ERROR: $TEMP_EXPORT_PATH/model.mar not found!"
    rm -rf "$TEMP_EXPORT_PATH"
    exit 1
fi
echo "$TEMP_EXPORT_PATH/model.mar created successfully."

echo "Step 3: Moving model.mar to its final location..."
mv "$TEMP_EXPORT_PATH/model.mar" "src/ml_model/model.mar"

if [ ! -f "src/ml_model/model.mar" ]; then
    echo "ERROR: Failed to move model.mar to src/ml_model/model.mar!"
    rm -rf "$TEMP_EXPORT_PATH"
    exit 1
fi
echo "model.mar moved to src/ml_model/model.mar."

echo "Step 4: Cleaning up temporary export directory..."
rm -rf "$TEMP_EXPORT_PATH"
echo "Temporary directory removed."

echo "Model archiving process completed successfully."
