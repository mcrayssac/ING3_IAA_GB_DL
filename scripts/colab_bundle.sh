#!/usr/bin/env bash
# Prépare dans dist/ les fichiers à téléverser sur Colab (dépôt privé) :
# projet.ipynb, projet-code.zip (code, résultats, checkpoints, démo) et fer2013.zip.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/jupytext --to ipynb notebooks/projet.py
rm -rf dist
mkdir -p dist
cp notebooks/projet.ipynb dist/

code_paths=(src notebooks/projet.py SPEC.md training/logs training/checkpoints)
[ -d data/demo ] && code_paths+=(data/demo)
zip -q -r dist/projet-code.zip "${code_paths[@]}" -x '*/__pycache__/*' '*smoke*' '*.DS_Store' '*.zip'
zip -q -r dist/fer2013.zip data/train data/test -x '*.DS_Store'
ls -lh dist
