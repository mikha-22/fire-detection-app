#!/bin/bash

# Ensure you are in the project root directory (e.g., fire-detection-app/)
# cd /path/to/your/fire-detection-app

echo "Step 1: Generating model.pth..."
# Run the python script to save model.pth into src/ml_model/
python src/ml_model/fire_detection_model.py

# Check if model.pth was created successfully
if [ ! -f "src/ml_model/model.pth" ]; then
    echo "ERROR: src/ml_model/model.pth not found after running fire_detection_model.py!"
    exit 1
fi
echo "src/ml_model/model.pth generated successfully."

echo "Step 2: Archiving model into model.mar..."
# Define a temporary directory for torch-model-archiver output
TEMP_EXPORT_PATH="src/ml_model/model_store_temp"
mkdir -p "$TEMP_EXPORT_PATH" # Create if it doesn't exist

# Run torch-model-archiver
# Paths are relative to the project root where this script is run.
torch-model-archiver --model-name model \
                     --version 1.0 \
                     --model-file src/ml_model/fire_detection_model.py \
                     --serialized-file src/ml_model/model.pth \
                     --handler src/ml_model/handler.py \
                     --requirements-file src/ml_model/requirements.txt \
                     --export-path "$TEMP_EXPORT_PATH" \
                     --force

# Check if model.mar was created in the temp export path
if [ ! -f "$TEMP_EXPORT_PATH/model.mar" ]; then
    echo "ERROR: $TEMP_EXPORT_PATH/model.mar not found after running torch-model-archiver!"
    rm -rf "$TEMP_EXPORT_PATH" # Clean up temp directory on failure
    exit 1
fi
echo "$TEMP_EXPORT_PATH/model.mar created successfully."

echo "Step 3: Moving model.mar to its final location for Docker..."
# Move the created model.mar to the location the Dockerfile expects
mv "$TEMP_EXPORT_PATH/model.mar" "src/ml_model/model.mar"

# Check if the move was successful
if [ ! -f "src/ml_model/model.mar" ]; then
    echo "ERROR: Failed to move model.mar to src/ml_model/model.mar!"
    rm -rf "$TEMP_EXPORT_PATH" # Clean up temp directory
    exit 1
fi
echo "model.mar moved to src/ml_model/model.mar."

echo "Step 4: Cleaning up temporary export directory..."
rm -rf "$TEMP_EXPORT_PATH"
echo "Temporary directory $TEMP_EXPORT_PATH removed."

echo "Model archiving process completed successfully."
echo "You can now build your Docker image."
