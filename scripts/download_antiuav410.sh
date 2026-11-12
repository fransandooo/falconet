#!/usr/bin/env bash
# Download and verify the Anti-UAV410 thermal-infrared drone tracking benchmark.
#
#   scripts/download_antiuav410.sh            # download + verify + extract
#   scripts/download_antiuav410.sh --verify   # only check an existing download
#
# Source: https://github.com/HwangBo94/Anti-UAV410 (official toolkit; Google Drive
# mirror of the archive). Paper: Huang et al., "Anti-UAV410: A Thermal Infrared
# Benchmark and Customized Scheme for Tracking Drones in the Wild", T-PAMI 2023.
#
# Licence: the dataset is published for research use and the official repository
# has no licence file. Never redistribute the data or derived weights; cite the paper.
#
# Data goes to $FALCONET_DATA_DIR/antiuav410 (mise sets FALCONET_DATA_DIR=~/datasets
# inside the repo); nothing is written to the repository.
set -euo pipefail

DATA_DIR="${FALCONET_DATA_DIR:-$HOME/datasets}/antiuav410"
ARCHIVE="$DATA_DIR/Anti-UAV410.zip"
URL="https://drive.usercontent.google.com/download?id=1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR&export=download&confirm=t"

# The authors publish no checksum. The size comes from the server; the SHA-256 was
# recorded from the first verified download (trust on first use) and pins the
# exact archive every later download must match.
EXPECTED_BYTES=9361681896
EXPECTED_SHA256="${ANTIUAV410_SHA256:-}"

log() { printf '[antiuav410] %s\n' "$*"; }

verify() {
    local size
    size=$(stat -c %s "$ARCHIVE")
    if [[ "$size" != "$EXPECTED_BYTES" ]]; then
        log "size mismatch: got $size bytes, expected $EXPECTED_BYTES"
        return 1
    fi
    log "size OK ($size bytes); computing SHA-256..."
    local sha
    sha=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
    if [[ -z "$EXPECTED_SHA256" ]]; then
        log "SHA-256: $sha (no pinned value yet: record it in this script)"
    elif [[ "$sha" != "$EXPECTED_SHA256" ]]; then
        log "SHA-256 mismatch: got $sha, expected $EXPECTED_SHA256"
        return 1
    else
        log "SHA-256 OK"
    fi
}

mkdir -p "$DATA_DIR"

if [[ "${1:-}" == "--verify" ]]; then
    verify
    exit
fi

is_zip() { [[ "$(head -c 4 "$1" | od -An -tx1 | tr -d ' ')" == "504b0304" ]]; }

# A partial download that isn't a zip is an error page (e.g. Google Drive's
# "Quota exceeded"): resuming on top of it would corrupt the archive.
if [[ -f "$ARCHIVE" ]] && ! is_zip "$ARCHIVE"; then
    rm -f "$ARCHIVE"
fi

if [[ ! -f "$ARCHIVE" || $(stat -c %s "$ARCHIVE") -lt $EXPECTED_BYTES ]]; then
    log "downloading to $ARCHIVE (9.4 GB, resumable: re-run to continue)"
    curl --fail --location --continue-at - --retry 5 --retry-delay 10 \
        --output "$ARCHIVE" "$URL"
    if ! is_zip "$ARCHIVE"; then
        log "the server returned a web page instead of the archive:"
        sed 's/<[^>]*>/ /g' "$ARCHIVE" | grep -oE "Quota exceeded|Virus scan[^.]*|Sign in" | head -n 1
        log "Google Drive limits downloads of popular public files per day; retry later,"
        log "or download it in a browser and put it at $ARCHIVE"
        rm -f "$ARCHIVE"
        exit 2
    fi
fi

verify

if [[ ! -d "$DATA_DIR/train" ]]; then
    log "extracting..."
    unzip -q "$ARCHIVE" -d "$DATA_DIR/extract-tmp"
    # Flatten a possible top-level folder so splits end up at $DATA_DIR/{train,val,test}.
    inner=$(find "$DATA_DIR/extract-tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [[ -d "$inner/train" ]]; then
        mv "$inner"/* "$DATA_DIR"/
    else
        mv "$DATA_DIR/extract-tmp"/* "$DATA_DIR"/
    fi
    rm -rf "$DATA_DIR/extract-tmp"
fi

for split in train val test; do
    count=$(find "$DATA_DIR/$split" -mindepth 1 -maxdepth 1 -type d | wc -l)
    log "$split: $count sequences"
done
log "done: $DATA_DIR"
