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
   region, with a per-region consecutive-deferral cap; backpressure or the
   global deadlock limit may force progress sooner;
7. optionally freezes the predicted terminal region while a region to its left
   remains unfinished (`regional_tail_guard=true`).

Three termination variants are exposed for attribution:

- `regional`: coarse predicted-terminal-region guard;
- `regional_filter`: while earlier regions are unfinished, exclude stop-token
  proposals and proposals below the stop confidence threshold from the
  terminal region; once a stop is accepted, ignore its masked suffix;
- `regional_defer`: defer the terminal region's entire scheduled update when
  that update contains a stop proposal. This is retained as a negative
  ablation, not the recommended operating point.

The released Fast-dLLM confidence-threshold algorithm is not modified and is
the direct control. The regional path intentionally rejects `use_cache=true`:
Fast-dLLM's prefix/dual cache assumes a sequential active block, whereas this
pilot can change tokens in many regions after every full-canvas forward.

## Mechanism telemetry

Every regional `generation_stats` record includes four diagnostic sections:

- `startup_mechanism`: every zero-quota, deferred, confidence-passed,
  gap-forced, and global-deadlock-forced bootstrap update, plus per-region
  first-commit/startup-completion NFEs and the minimum raw top-1 probability
  that controlled each decision;
- `concurrency_mechanism`: how many distinct regions actually committed on
  each forward and how often no region or multiple regions committed;
- `progress_balance`: observed adjacent revealed-token gaps;
- `commit_confidence`: the mean raw top-1 probability of startup,
  post-startup, and all committed tokens, including the fractions below the
  regional deferral threshold and above Fast-dLLM's `0.9` reference threshold.

The detailed `startup_mechanism.attempts` list is deliberately restricted to
the configured bootstrap window. It records repeated attempts while a region
still has fewer than the configured number of revealed tokens; it does not
confuse repeated deferrals with distinct tokens.

Summarize one or more run logs without printing prompts or responses:

```bash
python scripts/summarize_regional_mechanism.py logs/regional_filter_50.log
```

The run script's documented leading defaults are 32-token regions, 32 local
steps, a four-token maximum progress gap, a `0.4` startup threshold, a
two-token bootstrap window, and at most four consecutive deferrals per region
before that region is forced. They can be overridden without editing code. Use
`RUN_TAG` whenever changing settings so lm-eval output directories do not
collide:

```bash
REGION_SIZE=40 \
LOCAL_STEPS=32 \
MAX_PROGRESS_GAP=8 \
DEFERRAL_THRESHOLD=0.4 \
DEFERRAL_UNTIL_REVEALED=4 \
MAX_REGION_DEFERRALS=4 \
RUN_TAG=r40_d4_g8 \
LIMIT=50 \
bash scripts/run_gsm8k_regional.sh regional_filter
```

The expanded `40/4/8` setting is an ablation transferred from the earlier
pilot, not a verified Fast-dLLM-protocol winner. Keep it separate from the
documented `32/2/4` default until a paired validation run establishes that it
transfers to Dream Base with the five-shot prompt.

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

For the higher-accuracy terminal variant used in the leading prior
large-sample comparison, smoke-test `regional_filter` as well:

```bash
mkdir -p logs
LIMIT=2 bash scripts/run_gsm8k_regional.sh regional_filter \
  > logs/regional_filter_2.log 2>&1
python scripts/summarize_regional_mechanism.py logs/regional_filter_2.log
```

Then run the matched 50-example Fast-dLLM paper protocol separately for each
decoder:

```bash
LIMIT=50 bash scripts/run_gsm8k_regional.sh vanilla
LIMIT=50 bash scripts/run_gsm8k_regional.sh fast
LIMIT=50 bash scripts/run_gsm8k_regional.sh fast_cache
LIMIT=50 bash scripts/run_gsm8k_regional.sh regional
LIMIT=50 bash scripts/run_gsm8k_regional.sh regional_filter
LIMIT=50 bash scripts/run_gsm8k_regional.sh regional_defer
```

`fast` isolates Fast-dLLM's global confidence selector without caching.
`fast_cache` is the released dual-cache plus parallel path. The latter is an
important end-to-end comparison, but it is not a selector-only attribution
because the regional algorithm cannot safely reuse that sequential-block cache.

For a matched HumanEval comparison, the wrapper switches to zero-shot
HumanEval, enables code-evaluation confirmation and `escape_until`, and keeps
the model/decoder settings otherwise identical:

```bash
LIMIT=20 bash scripts/run_humaneval_regional.sh fast
LIMIT=20 bash scripts/run_humaneval_regional.sh regional_filter
```

HumanEval execution scoring still uses the repository's upstream
`postprocess_code.py` on each generated `samples_humaneval_*.jsonl` file.

This protocol uses Dream-v0-Base-7B and GSM8K 5-shot, matching Fast-dLLM's
released Dream guide. The non-cache paths print `generation_stats` per example
and end with a `generation_summary` containing mean NFE, synchronized
model-generation seconds, and canvas tokens/second. Fast-dLLM's unchanged cache
path retains its upstream timing output. Task accuracy remains in lm-eval's
normal result file/table.

No checkpoint or evaluation has been run in the local development workspace.
