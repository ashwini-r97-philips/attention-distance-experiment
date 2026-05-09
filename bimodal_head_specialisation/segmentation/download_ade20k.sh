#!/usr/bin/env bash
# Download ADE20K dataset (~924MB)
set -euo pipefail

DATA_DIR="${1:-/sudarshana/data}"
URL="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ZIP_FILE="$DATA_DIR/ADEChallengeData2016.zip"
DEST_DIR="$DATA_DIR/ADEChallengeData2016"

if [[ -d "$DEST_DIR/images/training" ]]; then
    echo "ADE20K already exists at $DEST_DIR"
    exit 0
fi

mkdir -p "$DATA_DIR"
echo "Downloading ADE20K to $DATA_DIR ..."
wget --no-check-certificate -c -O "$ZIP_FILE" "$URL"
echo "Extracting..."
unzip -q -o "$ZIP_FILE" -d "$DATA_DIR"
rm -f "$ZIP_FILE"
echo "Done. Dataset at: $DEST_DIR"
echo "  Training images: $(ls $DEST_DIR/images/training/ | wc -l)"
echo "  Validation images: $(ls $DEST_DIR/images/validation/ | wc -l)"
