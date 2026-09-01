#!/usr/bin/env bash
# Build the MiRA 2026 GNN deck.
#   ./compile.sh                 -> gnn_talk.pdf
#   ./compile.sh gnn_talk.tex    -> same
#   ENGINE=xelatex ./compile.sh  -> use xelatex instead of tectonic
#
# A XeTeX-based engine is required: the deck uses fontspec + a CJK font
# ("Noto Sans CJK TC") for the speaker's name. pdflatex will NOT work.
set -euo pipefail
cd "$(dirname "$0")"
TEX="${1:-gnn_talk.tex}"
ENGINE="${ENGINE:-}"

if [ -z "$ENGINE" ]; then
  if command -v tectonic >/dev/null 2>&1; then ENGINE=tectonic
  elif command -v xelatex >/dev/null 2>&1; then ENGINE=xelatex
  elif command -v lualatex >/dev/null 2>&1; then ENGINE=lualatex
  else echo "No XeTeX-capable engine found (tectonic/xelatex/lualatex)." >&2; exit 1
  fi
fi

case "$ENGINE" in
  tectonic) tectonic "$TEX" ;;
  *)        "$ENGINE" -interaction=nonstopmode "$TEX"
            "$ENGINE" -interaction=nonstopmode "$TEX" ;;
esac
echo "-> ${TEX%.tex}.pdf"
