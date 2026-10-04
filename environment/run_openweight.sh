#!/bin/zsh
# E9 on open-weight models (cross-model generality): 10 skewed + 10 balanced each, 6 in parallel, resumable.
cd "$(dirname $0)"
for m in deepseek/deepseek-v4-pro qwen/qwen3-235b-a22b-thinking-2507 openai/gpt-oss-120b; do
  short=${m##*/}
  for s in $(seq 0 9); do for v in skewed balanced; do
    [[ -f results/${short}_${v}_s${s}.json ]] || echo "$m $v $s"
  done; done
done | xargs -P 6 -L 1 zsh -c 'uv run -q --with numpy python e9_run.py --model $0 --variant $1 --seed $2 || echo FAIL $0 $1 $2'
echo OPENWEIGHT_DONE
