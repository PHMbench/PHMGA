#!/bin/bash

set -euo pipefail

echo "Configuring PHMGA shell wrappers..."
find scripts/sh -name "*.sh" -type f -exec chmod +x {} \;

pilot_count=$(find scripts/sh/pilot -name "*.sh" -type f | wc -l)
main_count=$(find scripts/sh/main -name "*.sh" -type f | wc -l)
ablation_count=$(find scripts/sh/ablation -name "*.sh" -type f | wc -l)
total_count=$((pilot_count + main_count + ablation_count))

echo "Wrapper permissions updated."
echo "Total wrappers: $total_count"
echo "  pilot: $pilot_count"
echo "  main: $main_count"
echo "  ablation: $ablation_count"
echo
echo "Authoritative experiment instructions remain in:"
echo "  doc/experiments/00_manual_runbook.md"
