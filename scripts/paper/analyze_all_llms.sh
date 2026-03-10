#!/bin/bash
# Analysis script to run after all LLM matrix experiments complete
# Usage: bash scripts/paper/analyze_all_llms.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/save/paper_matrix"

echo "=== LLM Matrix Analysis Script ==="
echo "Checking completion status..."

# Expected: 3 LLMs × 2 datasets × 3 ablations = 18 combinations
EXPECTED_PER_LLM=6
total_complete=0
total_records=0

llm_dirs=("m1_gemini25" "m2_gemini3" "m3_glm47")

# Check each LLM
for dir in "${llm_dirs[@]}"; do
    manifest="$OUTPUT_ROOT/$dir/manifest.jsonl"
    if [[ -f "$manifest" ]]; then
        count=$(wc -l < "$manifest")
        total_records=$((total_records + count))
        if [[ "$count" -ge "$EXPECTED_PER_LLM" ]]; then
            echo "✓ $dir: $count/$EXPECTED_PER_LLM records (COMPLETE)"
            total_complete=$((total_complete + 1))
        else
            echo "⏳ $dir: $count/$EXPECTED_PER_LLM records (running)"
        fi
    else
        echo "⏳ $dir: no manifest yet (starting)"
    fi
done

echo ""
echo "Total records: $total_records/18"

# Merge all manifests
echo ""
echo "=== Merging all manifests ==="
all_manifest="$OUTPUT_ROOT/all_llm_manifest.jsonl"
> "$all_manifest"
for dir in "${llm_dirs[@]}"; do
    manifest="$OUTPUT_ROOT/$dir/manifest.jsonl"
    if [[ -f "$manifest" ]]; then
        cat "$manifest" >> "$all_manifest"
    fi
done
echo "Created: $all_manifest ($(wc -l < "$all_manifest") records)"

# Run collection scripts
echo ""
echo "=== Collecting results ==="
conda run -n agent python "$REPO_ROOT/scripts/paper/collect_matrix_results.py" \
    --manifest "$all_manifest" \
    --output-dir "$OUTPUT_ROOT"

# Run analysis scripts
echo ""
echo "=== Generating analysis drafts ==="
conda run -n agent python "$REPO_ROOT/scripts/paper/generate_analysis_draft.py" \
    --manifest "$all_manifest" \
    --output-dir "$OUTPUT_ROOT"

# Per-LLM analysis
echo ""
echo "=== Per-LLM analysis ==="
for dir in "${llm_dirs[@]}"; do
    manifest="$OUTPUT_ROOT/$dir/manifest.jsonl"
    if [[ -f "$manifest" ]] && [[ -s "$manifest" ]]; then
        echo "Analyzing $dir..."
        conda run -n agent python "$REPO_ROOT/scripts/paper/collect_matrix_results.py" \
            --manifest "$manifest" \
            --output-dir "$OUTPUT_ROOT/$dir"

        conda run -n agent python "$REPO_ROOT/scripts/paper/generate_analysis_draft.py" \
            --manifest "$manifest" \
            --output-dir "$OUTPUT_ROOT/$dir"
    fi
done

# Summary
echo ""
echo "=== Analysis Complete ==="
echo "Generated files:"
ls -la "$OUTPUT_ROOT"/*.csv "$OUTPUT_ROOT"/*.md 2>/dev/null | tail -5

echo ""
echo "Main results: $OUTPUT_ROOT/paper_main_results.csv"
echo "Analysis draft: $OUTPUT_ROOT/analysis_draft.md"
