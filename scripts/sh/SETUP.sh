#!/bin/bash

set -euo pipefail

echo "Configuring PHMGA shell wrappers..."
find scripts/sh -name "*.sh" -type f -exec chmod +x {} \;

pilot_count=$(find scripts/sh/pilot -name "*.sh" -type f | wc -l)
main_count=$(find scripts/sh/main -name "*.sh" -type f | wc -l)
ablation_count=$(find scripts/sh/ablation -name "*.sh" -type f | wc -l)
helper_count=$(find scripts/sh -maxdepth 1 -name "_common.sh" -type f | wc -l)
experiment_wrapper_count=$((pilot_count + main_count + ablation_count))
total_shell_files=$((experiment_wrapper_count + helper_count))

echo "Wrapper permissions updated."
echo "Experiment wrappers: $experiment_wrapper_count"
echo "  pilot: $pilot_count"
echo "  main: $main_count"
echo "  ablation: $ablation_count"
echo "Shared helpers (not experiment wrappers): $helper_count"
echo "Total shell files under scripts/sh: $total_shell_files"
echo
echo "Wrapper behavior summary:"
echo "  pilot -> forces offline_stub"
echo "  main -> defaults to current wrapper tuple (codex_cli / gpt-5.3-codex)"
echo "  method ablations -> default to the same tuple unless selected backend requires override"
echo "  ablation/provider -> Stage B backend comparison wrappers"
echo
echo "Authoritative experiment instructions remain in:"
echo "  doc/experiments/00_manual_runbook.md"
