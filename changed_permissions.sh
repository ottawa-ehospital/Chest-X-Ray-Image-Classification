#!/bin/sh

# Debugging  message
echo "Executing changed_permissions.sh script..."

# Debugging message
echo "Current directory: $(pwd)"

# Attempt to change permissions
chmod a+rx ./chest_xray_classification_model_20240407_071036/saved_model.pb

# Check if the chmod command succeeded
if [ $? -eq 0 ]; then
    echo "Permissions changed successfully."
else
    echo "Failed to change permissions."
    exit 1  # Exit with error status
fi
