#!/bin/zsh
# E9 master run (resumable: finished calls come from results/cache.jsonl). 10 in parallel.
#   skewed 0-19, balanced 0-19 (control), skewed_neural 0-11, skewed_purpose 0-11.
cd "$(dirname $0)"
{ for s in $(seq 0 19); do echo "skewed $s"; echo "balanced $s"; done
  for s in $(seq 0 11); do echo "skewed_neural $s"; echo "skewed_purpose $s"; done; } \
 | while read v s; do [[ -f results/grok_${v}_s${s}.json ]] || echo "$v $s"; done \
 | xargs -P 10 -L 1 zsh -c 'uv run -q --with numpy python e9_run.py --variant $0 --seed $1 || echo FAIL $0 $1'
echo MASTER_DONE
