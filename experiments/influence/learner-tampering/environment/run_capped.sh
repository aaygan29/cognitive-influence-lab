#!/bin/zsh
# E9 defended variant: amplitude cap + curriculum band, 12 Grok episodes, 4 in parallel (resumable).
cd "$(dirname $0)"
for s in $(seq 0 11); do [[ -f results/grok_skewed_capped_s${s}.json ]] || echo "skewed_capped $s"; done \
 | xargs -P 4 -L 1 zsh -c 'uv run -q --with numpy python e9_run.py --variant $0 --seed $1 || echo FAIL $0 $1'
echo CAPPED_DONE
