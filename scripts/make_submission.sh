#!/bin/bash
# Build the submission zip ON GIANO (where data/, results/ and traces/ live).
#   bash make_submission.sh [PROJECT_DIR]          default PROJECT_DIR=/scratch.hpc/$USER/project
# Expects README.md (and optionally notebook/) inside PROJECT_DIR; creates PROJECT_DIR/../submission/bigdata_project.zip
# Not included on purpose: FAISS indices / embeddings (rebuilt by 03_retrieval_eval.py), model caches, the venv,
# smoke-test files, SLURM logs, personal notes (PROGRESS.md / KNOWLEDGE.md).
set -e
P=${1:-/scratch.hpc/$USER/project}
OUT=$(dirname "$P")/submission
S=$OUT/bigdata_project
rm -rf "$S" "$OUT/bigdata_project.zip"; mkdir -p "$S"/{scripts,data,results,traces,cache,env}

cp "$P"/README.md "$S"/ 2>/dev/null || echo "WARNING: README.md missing in $P"
[ -d "$P/notebook" ] && cp -r "$P/notebook" "$S"/notebook || echo "note: no notebook/ yet"
cp "$P"/scripts/*.py "$P"/scripts/*.sbatch "$S"/scripts/ 2>/dev/null || true
cp "$P"/data/*.jsonl "$S"/data/
# results: tables, figures, analysis logs; skip smoke tests
find "$P/results" -maxdepth 1 -type f ! -name '*smoke*' -exec cp {} "$S"/results/ \;
[ -d "$P/results/holdout" ] && cp -r "$P/results/holdout" "$S"/results/holdout
# traces: gzip, skip smoke tests
for f in "$P"/traces/*.jsonl; do case "$f" in *smoke*) continue;; esac; gzip -c "$f" > "$S/traces/$(basename "$f").gz"; done
cp "$P"/cache/*.jsonl "$S"/cache/ 2>/dev/null || true
if [ -f "$P/env/requirements.txt" ]; then cp "$P/env/requirements.txt" "$S/env/requirements.txt"; else pip freeze > "$S/env/requirements.txt" 2>/dev/null || true; fi; [ -s "$S/env/requirements.txt" ] || { rm -f "$S/env/requirements.txt"; echo "!!! pip freeze produced nothing: activate the venv and rerun, README points to env/requirements.txt"; }

# --- safety checks -----------------------------------------------------------------------------------------
echo "--- scanning for secrets ---"
if grep -rIlE "(hf_[A-Za-z0-9]{20,}|NCBI_API_KEY[^0-9a-fA-F]{0,8}[0-9a-fA-F]{20,})" "$S" 2>/dev/null; then
  echo "!!! possible secret found in the files listed above — remove it before sending"; exit 1
else echo "no obvious secrets found"; fi

(cd "$OUT" && zip -qr bigdata_project.zip bigdata_project)
echo "--- contents ---"; du -sh "$S"/* | sort -h
SIZE=$(du -m "$OUT/bigdata_project.zip" | cut -f1)
echo "ZIP: $OUT/bigdata_project.zip  (${SIZE} MB)"
[ "$SIZE" -gt 18 ] && echo "!!! larger than ~18 MB: university mailboxes usually cap attachments near 20 MB — share an OneDrive link instead, or drop traces/"
exit 0
