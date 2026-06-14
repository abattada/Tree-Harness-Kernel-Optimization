#!/usr/bin/env bash
# Regenerate every table, figure, and the PDF from the run journals.
# No hand-entered numbers: aggregate.py / token_report.py / make_data_figs.py
# all read runs/*/journal.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."          # project root
PY="${PY:-python}"

DS_RUNS=(runs/minimal_kb_0614_091650 runs/minimal_kb_0614_093601 runs/minimal_kb_0614_120740)
PRO_RUN=runs/pilot_full_pro_0614_124354     # DeepSeek-v4-pro
AGENT=runs/agent_int4_gemm

echo "== aggregate (DeepSeek flash + pro + Claude agent) =="
$PY analysis/aggregate.py --runs "${DS_RUNS[@]}" "$PRO_RUN" "$AGENT" --out report/tables/compare

echo "== token cost =="
$PY analysis/token_report.py --runs "${DS_RUNS[@]}" | tee report/tables/cost.txt

echo "== data figures =="
$PY report/make_data_figs.py

echo "== appendix raw materials (verbatim transcript: rope 4.29x candidate) =="
mkdir -p report/appendix
TX=runs/minimal_kb_0614_120740/transcripts/0004_1c61323faaee_generation.json
$PY - "$TX" <<'PYEOF'
import json, sys
t = json.loads(open(sys.argv[1]).read())
open('report/appendix/system_prompt.txt', 'w').write(t['system'])
open('report/appendix/input_prompt.txt', 'w').write(t['prompt'])
open('report/appendix/agent_output.txt', 'w').write(t['response'])
print('  system %d / prompt %d / response %d chars'
      % (len(t['system']), len(t['prompt']), len(t['response'])))
PYEOF
cp runs/agent_int4_gemm/best.py report/appendix/claude_int4gemm_best.py

echo "== compile PDF =="
cd report && latexmk -pdf -interaction=nonstopmode REPORT.tex >/dev/null
echo "wrote report/REPORT.pdf"
