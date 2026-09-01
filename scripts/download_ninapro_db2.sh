#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/lp-dev/users/nvidia/EMG_Diffusion_data/raw/ninapro_db2"
ARCHIVES="$BASE/archives"
EXTRACTED="$BASE/extracted"
MANIFESTS="$BASE/manifests"

mkdir -p "$ARCHIVES" "$EXTRACTED" "$MANIFESTS/subjects"

download_one() {
    local subject="$1"
    local url="https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s${subject}.zip"
    local final="$ARCHIVES/DB2_s${subject}.zip"
    local partial="$final.part"
    local record="$MANIFESTS/subjects/s$(printf '%02d' "$subject").tsv"
    local record_tmp="$record.tmp"

    if [[ -f "$final" ]] && unzip -tq "$final" >/dev/null 2>&1; then
        printf '[s%02d] valid archive already present\n' "$subject"
    else
        if [[ -f "$final" ]]; then
            mv "$final" "$final.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
        fi
        printf '[s%02d] downloading\n' "$subject"
        curl -fL \
            --retry 8 \
            --retry-delay 3 \
            --retry-all-errors \
            --continue-at - \
            --output "$partial" \
            "$url"
        unzip -tq "$partial" >/dev/null
        mv "$partial" "$final"
        printf '[s%02d] download and ZIP test complete\n' "$subject"
    fi

    local bytes
    local digest
    bytes="$(stat -c '%s' "$final")"
    digest="$(sha256sum "$final" | awk '{print $1}')"
    printf '%02d\t%s\t%s\t%s\t%s\n' \
        "$subject" "$url" "$bytes" "$digest" "zip_test_pass" > "$record_tmp"
    mv "$record_tmp" "$record"
}

export BASE ARCHIVES EXTRACTED MANIFESTS
export -f download_one

printf 'Downloading official NinaPro DB2 preprocessed subject archives with two concurrent transfers.\n'
seq 1 40 | xargs -P 2 -n 1 bash -c 'download_one "$1"' _

{
    printf 'subject\turl\tbytes\tsha256\tarchive_status\n'
    find "$MANIFESTS/subjects" -maxdepth 1 -type f -name 's*.tsv' -print0 \
        | sort -z \
        | xargs -0 cat
} > "$MANIFESTS/download_manifest.tsv"

extract_one() {
    local subject="$1"
    local archive="$ARCHIVES/DB2_s${subject}.zip"
    local target="$EXTRACTED/DB2_s${subject}"
    local count=0

    if [[ -d "$target" ]]; then
        count="$(find "$target" -maxdepth 1 -type f -name '*.mat' | wc -l)"
    fi

    if [[ "$count" -eq 3 ]]; then
        printf '[s%02d] three MATLAB files already extracted\n' "$subject"
    else
        printf '[s%02d] extracting\n' "$subject"
        unzip -q -o "$archive" -d "$EXTRACTED"
        count="$(find "$target" -maxdepth 1 -type f -name '*.mat' | wc -l)"
        if [[ "$count" -ne 3 ]]; then
            printf '[s%02d] ERROR: expected 3 MATLAB files, found %s\n' "$subject" "$count" >&2
            return 1
        fi
        printf '[s%02d] extraction complete\n' "$subject"
    fi
}

export -f extract_one

printf 'Extracting verified archives with two concurrent workers.\n'
seq 1 40 | xargs -P 2 -n 1 bash -c 'extract_one "$1"' _

{
    printf 'relative_path\tbytes\n'
    find "$EXTRACTED" -type f -name '*.mat' -printf '%P\t%s\n' | sort -V
} > "$MANIFESTS/extracted_manifest.tsv"

archive_count="$(find "$ARCHIVES" -maxdepth 1 -type f -name 'DB2_s*.zip' | wc -l)"
mat_count="$(find "$EXTRACTED" -type f -name '*.mat' | wc -l)"
archive_bytes="$(find "$ARCHIVES" -maxdepth 1 -type f -name 'DB2_s*.zip' -printf '%s\n' | awk '{s+=$1} END{printf "%.0f", s}')"
mat_bytes="$(find "$EXTRACTED" -type f -name '*.mat' -printf '%s\n' | awk '{s+=$1} END{printf "%.0f", s}')"

{
    printf 'retrieved_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'archive_count=%s\n' "$archive_count"
    printf 'mat_file_count=%s\n' "$mat_count"
    printf 'archive_bytes=%s\n' "$archive_bytes"
    printf 'mat_bytes=%s\n' "$mat_bytes"
} > "$MANIFESTS/completion_summary.txt"

if [[ "$archive_count" -ne 40 || "$mat_count" -ne 120 ]]; then
    printf 'ERROR: expected 40 archives and 120 MATLAB files; found %s and %s.\n' \
        "$archive_count" "$mat_count" >&2
    exit 1
fi

printf 'DB2 download complete: %s verified archives and %s MATLAB files.\n' \
    "$archive_count" "$mat_count"
