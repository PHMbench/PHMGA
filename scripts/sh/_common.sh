#!/bin/bash

phmga_find_root() {
  local start_dir="$1"
  local dir="$start_dir"
  while [ "$dir" != "/" ] && { [ ! -f "$dir/main.py" ] || [ ! -f "$dir/README.md" ]; }; do
    dir="$(cd "$dir/.." && pwd)"
  done
  if [ ! -f "$dir/main.py" ]; then
    echo ""
    return 1
  fi
  echo "$dir"
}

phmga_enter_repo() {
  local start_dir="$1"
  local root_dir
  root_dir="$(phmga_find_root "$start_dir")"
  if [ -z "$root_dir" ]; then
    echo "Repository root not found." >&2
    exit 1
  fi
  cd "$root_dir"
}

phmga_assert_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

phmga_default_model_for_provider() {
  local provider="$1"
  case "$provider" in
    codex_cli)
      echo "gpt-5.3-codex"
      ;;
    openai)
      echo "gpt-5.3-codex"
      ;;
    openrouter)
      echo "z-ai/glm-4.5-air:free"
      ;;
    *)
      echo ""
      ;;
  esac
}

phmga_prepare_optional_provider() {
  local default_provider="$1"
  local default_model="$2"
  local provider="${PHMGA_LLM_PROVIDER:-}"
  local model="${PHMGA_LLM_MODEL:-}"
  declare -ga PHMGA_PROVIDER_ARGS=()
  if [ -z "$provider" ]; then
    if [ -n "$model" ]; then
      echo "PHMGA_LLM_MODEL requires PHMGA_LLM_PROVIDER." >&2
      exit 1
    fi
    PHMGA_SELECTED_PROVIDER="preset"
    PHMGA_SELECTED_MODEL="preset"
    return
  fi
  if [ -z "$model" ]; then
    if [ "$provider" = "$default_provider" ]; then
      model="$default_model"
    else
      model="$(phmga_default_model_for_provider "$provider")"
    fi
  fi
  PHMGA_SELECTED_PROVIDER="$provider"
  PHMGA_SELECTED_MODEL="$model"
  PHMGA_PROVIDER_ARGS=(llm.mode=provider "llm.provider=$provider" "llm.model=$model")
}

phmga_prepare_required_provider() {
  local default_provider="$1"
  local default_model="$2"
  local provider="${PHMGA_LLM_PROVIDER:-$default_provider}"
  local model="${PHMGA_LLM_MODEL:-$default_model}"
  declare -ga PHMGA_PROVIDER_ARGS=()
  PHMGA_SELECTED_PROVIDER="$provider"
  PHMGA_SELECTED_MODEL="$model"
  PHMGA_PROVIDER_ARGS=(llm.mode=provider "llm.provider=$provider" "llm.model=$model")
}

phmga_prepare_fixed_provider() {
  local fixed_provider="$1"
  local default_model="$2"
  local model="${PHMGA_LLM_MODEL:-$default_model}"
  declare -ga PHMGA_PROVIDER_ARGS=()
  PHMGA_SELECTED_PROVIDER="$fixed_provider"
  PHMGA_SELECTED_MODEL="$model"
  PHMGA_PROVIDER_ARGS=(llm.mode=provider "llm.provider=$fixed_provider" "llm.model=$model")
}

phmga_log_provider_choice() {
  if [ "${PHMGA_SELECTED_PROVIDER:-preset}" = "preset" ]; then
    echo "LLM provider/model: inherited from preset"
    return
  fi
  echo "LLM provider: $PHMGA_SELECTED_PROVIDER"
  echo "LLM model: $PHMGA_SELECTED_MODEL"
}
