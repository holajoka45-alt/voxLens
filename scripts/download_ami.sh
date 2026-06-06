#!/bin/bash
# Download AMI Corpus (Mix Headset partition)
# Requires: wget, ~50GB disk space
#
# Usage: bash scripts/download_ami.sh /path/to/output

set -euo pipefail

OUTPUT_DIR="${1:-./data/ami}"
AMI_URL="http://groups.inf.ed.ac.uk/ami"

echo "=== Downloading AMI Corpus ==="
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Download annotations
echo "Downloading RTTM annotations..."
wget -q --show-progress -O "$OUTPUT_DIR/ami_manual_annotations.rttm" \
    "$AMI_URL/download/temporarilyUnavaliable/ami_manual_annotations.rttm" || \
    echo "Warning: Could not download annotations. Check URL."

echo ""
echo "Done. Audio files need to be downloaded separately from:"
echo "  https://groups.inf.ed.ac.uk/ami/download/"
echo ""
echo "After downloading, place audio in: $OUTPUT_DIR/audio/"
