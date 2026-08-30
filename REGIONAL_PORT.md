# Balanced regional decoding port

This branch starts from NVLabs Fast-dLLM commit
`a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` and adds one Dream algorithm,
`regional_balanced`.

The model forward is unchanged: Dream still evaluates the complete prompt and
masked generation canvas with full visibility. Only the post-forward token
selection policy changes.

For a 256-token response canvas and 32-token regions, the algorithm:

1. builds eight fixed contiguous response regions;
2. evaluates Dream's linear transfer quota on a separate local schedule for
   each region;
3. chooses the highest-confidence candidates *inside each scheduled region*
   using the configured ordinary Dream policy (entropy by default);
4. measures progress as actually revealed tokens, never scheduled iterations;
5. prevents a child from leading its left neighbor and prevents a parent from
   leading its child by the configured token gap;
6. permits low-confidence deferral only for the first configured reveals in a
   region, unless backpressure or the global deadlock limit forces progress;
7. optionally freezes the predicted terminal region while a region to its left
   remains unfinished (`regional_tail_guard=true`).

The released Fast-dLLM confidence-threshold algorithm is not modified and is
the direct control. The regional path intentionally rejects `use_cache=true`:
Fast-dLLM's prefix/dual cache assumes a sequential active block, whereas this
pilot can change tokens in many regions after every full-canvas forward.

## Vast setup

From the repository root:

```bash
cd v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Fast-dLLM does not pin torch.  Pin a CUDA 12.1 wheel that is compatible with
# the 570-series drivers used by the target Vast image.
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
cd dream
```

First run the two-example smoke test:

```bash
LIMIT=2 bash scripts/run_gsm8k_regional.sh regional
```

Then run the matched 50-example Fast-dLLM paper protocol separately for each
decoder:

```bash
LIMIT=50 bash scripts/run_gsm8k_regional.sh vanilla
LIMIT=50 bash scripts/run_gsm8k_regional.sh fast
LIMIT=50 bash scripts/run_gsm8k_regional.sh fast_cache
LIMIT=50 bash scripts/run_gsm8k_regional.sh regional
```

`fast` isolates Fast-dLLM's global confidence selector without caching.
`fast_cache` is the released dual-cache plus parallel path. The latter is an
important end-to-end comparison, but it is not a selector-only attribution
because the regional algorithm cannot safely reuse that sequential-block cache.

This protocol uses Dream-v0-Base-7B and GSM8K 5-shot, matching Fast-dLLM's
released Dream guide. The non-cache paths print `generation_stats` per example
and end with a `generation_summary` containing mean NFE, synchronized
model-generation seconds, and canvas tokens/second. Fast-dLLM's unchanged cache
path retains its upstream timing output. Task accuracy remains in lm-eval's
normal result file/table.

No checkpoint or evaluation has been run in the local development workspace.
