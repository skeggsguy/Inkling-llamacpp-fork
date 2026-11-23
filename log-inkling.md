# Inkling-2 Local Fixes & PR Prep Log

Repo: `/Users/tomskeggs/dev/llama.cpp-inkling`
Branch: `metal-banded` (based on `b12588-mix` tag `b12588-mix-c0c3de8-src` @ `5b0fb9408` + local PR 1 Metal banded commits through `ac3acb5b8`).
Separate clean PR 2 worktree: `/Users/tomskeggs/dev/llama.cpp-inkling-pr2` (branch `metal-banded-q8`, stacked on PR #404 `58d6e4c5b` atop `origin/inkling2-attn-banded` `0edf1d668`).

---

## Commit Summary (`ac3acb5b8..HEAD`)

| Commit | Area | PR Target | Summary |
|---|---|---|---|
| `26ff0a2ce` | Metal banded FA | PR #404 (sync) | Sync local `metal-banded` branch with PR #404 cleanup (drop bf16 `rel` path, trim test matrix) |
| `aff373445` | Metal banded FA (`q8_0`/quant) | PR 2 (`unslothai/llama.cpp`) | Enable quantized K/V (`q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`) on Metal `FLASH_ATTN_EXT_BANDED` + `inkling2.cpp` |
| `a76edf505` | Metal FA buffer reserve | Standalone PR (`ggml-org` / `unslothai`) | Skip `extra_kv_f16` (when not quantized or `use_vec`) and `extra_rows` (when `use_vec`) |
| `39e7ab4ac` | Banded FA + SWA KV cache | Follow-up PR (`unslothai/llama.cpp`) | Active SWA window check + 2-run `ggml_concat` wrap-seam stitching for `FLASH_ATTN_EXT_BANDED` |
| `a77861652` | GGML `argmax` (DSpark uplift #1) | Standalone PR (`ggml-org` / `unslothai`) | Deterministic first-index tie-breaking in `GGML_OP_ARGMAX` across CPU, CUDA, Metal, Vulkan + tests |
| `8ad8d90f0` | DSpark (DSpark uplift #2) | Follow-up PR (`unslothai/llama.cpp`) | Precompute `dspark_w2_eff = target.output @ dspark_hidden_out`, simplify Markov chain argmax & unpadded vocab slicing |
| `bc05d21ce` | Banded FA + non-unified KV | Follow-up PR (`unslothai/llama.cpp`) | Reserve banded FA graph for multi-stream non-unified KV (`n_stream > 1`), dropping `3x200K` compute reserve from `20.4 GiB` -> `3.26 GiB` |
| `721e555b8` | Server scheduling | Optional PR / local tuning | Dynamic decode-priority chunked prefill (2 pure decode ticks per `128`-token prefill chunk while any slot writes; full `-b/-ub` when solo) |

---

## Detailed Notes per PR Candidate

### 1. PR 1: Metal `FLASH_ATTN_EXT_BANDED` (F16/F32/BF16) - PR #404 (Already Open)
- Local commits: `03d439592`, `9f8d364bf`, `6a65819c0`, `ac3acb5b8`, synced with `26ff0a2ce`.
- Note: PR #404 targets `unslothai/llama.cpp` branch `inkling2-attn-banded` (`#386`). Wait for #404 to merge/settle before opening PR 2.

### 2. PR 2: Metal Quantized K/V for `FLASH_ATTN_EXT_BANDED` (`aff373445`)
- **Clean PR branch already prepared in:** `/Users/tomskeggs/dev/llama.cpp-inkling-pr2` (`metal-banded-q8`, uncommitted so you can write your own commit/PR text).
- **Files:**
  - `ggml/src/ggml-metal/ggml-metal-device.m`: add `GGML_TYPE_Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` to `GGML_OP_FLASH_ATTN_EXT_BANDED` in `supports_op` (+5 lines).
  - `src/models/inkling2.cpp` (only needed on `b12588-mix`; `#386` already allows quantized KV types in `inkling2_loop.cpp`).
  - Optional tests in `llama.cpp-inkling-pr2/tests/test-backend-ops.cpp`: 6 extra quantized tensor-kernel eval cases + 4 Mini perf cases.
- **How it works:** Zero shader edits. Reuses Metal's existing `kv_f16` upfront dequant (`fa_aux.metal`) for `n_q >= 32` into `tensor`/`half8x8`, and `fa_q*.metal` (which includes `fa_common.metal`) for `n_q < 32`.
- **Verified numbers (M5 Pro, Inkling-2-Mini-UD-Q8_K_XL):**
  - `test-backend-ops -o FLASH_ATTN_EXT_BANDED -b MTL0`: 97/97 pass on both `tensor` and `GGML_METAL_TENSOR_DISABLE=1` (`half8x8`), including all 18 of Daniel's quantized cases.
  - Plain `FLASH_ATTN_EXT`: 5381/5381 pass on both kernels; 224-case ABBA perf ratio geomean `1.0072`, median `1.0023`.
  - `llama-fit-params -c 200000 -ub 512 -np 1 -ctk q8_0 -ctv q8_0`: KV+compute `12,743 MiB` (dense) -> `4,689 MiB` (banded, `-8,054 MiB` vs base Q8_0, `-3,054 MiB` vs F16 `7,743 MiB`). Q4_0 banded: `3,065 MiB`.
  - `llama-bench` pp2048 ABBA (`-d 4096,32768`): Q8_0 `353.6 -> 427.0 t/s` (1.21x at 4K) and `175.8 -> 374.7 t/s` (2.13x at 32K, 98.7% of F16 `379.5 t/s`).
  - KLD vs F16 (`-c 4096 --chunks 4`): Base Q8_0 dense `0.005406` (`98.083%` same top) vs Patched Q8_0 tensor `0.005599` (`98.095%`, within <0.5 SE).

### 3. PR 3 Candidate: Metal `use_vec` FA Reserve Blowout (`a76edf505`)
- **Files:** `ggml/src/ggml-metal/ggml-metal-ops.cpp` (`ggml_metal_op_flash_attn_ext_extra_kv_f16`, `ggml_metal_op_flash_attn_ext_extra_rows`).
- **Problem:** On unified KV caches with multiple slots (`-kvu -np 5 -c 1000000`), `llama-kv-cache.cpp` reserves worst-case `n_kv_rows = n_kv` x `n_slices = 5`. For small-query models like `dspark` (`ne01 = 8 < 20`, always runs `use_vec` and never uses `bid_kv_f16` or `bid_rows`), Metal still reserved `19,535 MiB` (`extra_rows`) + `3,907 MiB` (`extra_kv_f16`) = **`23,442 MiB`** of unused compute buffer for a `372 MiB` draft model, causing immediate Metal OOM on startup.
- **Fix:**
  - `extra_kv_f16`: return `0` when `!ggml_is_quantized(op->src[1]->type) || ggml_metal_op_flash_attn_ext_use_vec(op)`.
  - `extra_rows`: return `0` when `!kv_rows || ggml_metal_op_flash_attn_ext_use_vec(op)`.
- **Impact:** `dspark` MTL0 compute buffer dropped from `23,616 MiB` -> `174 MiB` (saves **~22.9 GiB** at 1M pool; `116 MiB` at 600K pool).
- **Upstream note:** Worth double-checking whether `extra_kv_f16` / `extra_rows` can ever transition from `use_vec == true` at reserve time to `use_vec == false` at runtime for a given graph (for draft models with `n_ubatch >= 20` on prefill, `sched_reserve` reserves both `pp` (`ne01 = n_ubatch`) and `tg` (`ne01 = n_seq`), taking the max across both graphs, so check `pp` vs `tg` reserve behavior cleanly before upstreaming to `ggml-org/llama.cpp`).

### 4. PR 4 Candidate: Banded FA SWA Window + 2-Run Wrap Stitching (`39e7ab4ac`)
- **Files:** `src/llama-graph.h`, `src/llama-graph.cpp`, `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/models/inkling2.cpp`.
- **Problem:**
  1. `get_n_kv_pos_contiguous()` demanded `pos = 0 .. pos_max` in one unbroken run from cell 0. For SWA layers (32/48 layers in Inkling-2-Mini, `n_swa = 1024`, ring buffer size `5,632-6,144` cells under `-np 5 -kvu`), the instant a prompt passed 5,632/6,144 tokens, the SWA ring wrapped and evicted `pos = 0`. All 32 SWA layers fell back to `kq_b != nullptr`, which disabled Flash Attention completely (`use_flash_attn = false` in `build_attn_mha`) and ran naive unfused attention over all `5,632` cells, dropping live prefill from `~580 t/s` to `~240 t/s`.
  2. Even after restricting the contiguity check to the active SWA window `[p_min .. pos_max]` (`p_min = pos_start - n_swa + 1`), 1 out of every 6 ubatches (`-ub 1024`, `6,144`-cell ring) still straddled the ring wrap boundary and spiked from `3.3 s` to `8.7 s` (`235 t/s`).
- **Fix:**
  - Replaced scalar `n_kv_banded` with `struct llama_kv_banded_span { c0, n0, c1, n1 }` (includes Daniel's `1ac4b1c91` non-zero `c0` unified-KV offset).
  - For standard SWA (`n_swa > 0`), checks only the active window `p_min .. pos_max` (verified with `seq_n_cells(seq_id, p_min) == n_span` so no active key sits outside).
  - When the active SWA window splits across the ring seam into 2 runs `[c0..c0+n0)` + `[c1..c1+n1)`, `inkling2.cpp` stitches `K`, `V`, and `mask` with `ggml_concat` (~`4.2 MiB` / layer in `q8_0`, < 1 ms total across 32 layers) and feeds directly into `ggml_flash_attn_ext_banded`.
  - Prepends 1 already-masked cell if `c0` is odd so `F16` mask view offsets stay 4-byte aligned on Metal.
  - Bonus: non-wrapped SWA layers slice `[c0..c0+n0)` (`2,048` keys) instead of `[0..c0+n0)` (up to `6,144` keys), and slots 1-4 skip scanning earlier slots' cells.
- **Verified numbers (live server, `-c 600000 -np 5 -kvu -ub 1024 -ctk q8_0 -ctv q8_0`):**
  - 12,047-token dual-needle test (needles at token 0 and right on the 6,144 wrap seam): both recalled cleanly.
  - Wrap-seam 2K chunk (`6,156 -> 8,204`): `8.73 s (235 t/s)` -> **`3.34 s (613 t/s)`** (zero dip).
  - Overall 12K prompt eval: `~236-276 t/s` -> **`601.6 t/s`** end-to-end (**`662.1 t/s`** raw), `0` changed-size graph reallocs.
- **Rebase note for `#386`:** On `unslothai/llama.cpp` `inkling2-attn-banded`, apply the model-side hunk to `src/models/inkling2_loop.cpp` (instead of `b12588-mix`'s `inkling2.cpp`) and verify non-Metal backends (CUDA/CPU `ggml_concat` support for quantized K/V types; if CUDA doesn't support quantized `GGML_OP_CONCAT`, gate `n1 > 0` on backend `supports_op(GGML_OP_CONCAT)`!).

### 5. PR 5 Candidate (DSpark Uplift #1): Deterministic First-Index `GGML_OP_ARGMAX` (`a77861652`)
- **Files:** `ggml/src/ggml-cpu/vec.h`, `ggml/src/ggml-cuda/argmax.cu`, `ggml/src/ggml-metal/kernels/misc.metal`, `ggml/src/ggml-vulkan/vulkan-shaders/argmax.comp`, `tests/test-backend-ops.cpp`.
- **Problem:** `GGML_OP_ARGMAX` had inconsistent tie-breaking across backends (CPU used `MAX` + `==` picking last index; Metal used `simd_max` on index picking last; CUDA/Vulkan had arbitrary or inconsistent tie order), forcing `dspark.cpp` to build a heavy workaround (`arange` + `pool_1d` + `repeat` + `sub` + `step` + `mul`) to guarantee first-index tie order.
- **Fix:** Standardized all four backends (CPU, CUDA, Metal `argmax`/`argmax_part`/`argmax_final`, Vulkan) to pick the smallest index on ties (init sentinel `-INFINITY` / `INT_MAX`, reduce via min-index on equal max, map `INT_MAX -> 0`), and added `ties = true` eval cases (`{1024, 8}`, `{10000, 8}`) to `test_argmax` in `tests/test-backend-ops.cpp`.
- **PR prep note:** Run `test-backend-ops test -o ARGMAX` on CPU + Metal (and CUDA/Vulkan CI) before submitting; check if other backends (SYCL/ROCm/OpenCL/etc.) need matching tie-break updates for `ggml-org/llama.cpp`.

### 6. PR 6 Candidate (DSpark Uplift #2): Precomputed `dspark_w2_eff` + Head Cleanup (`8ad8d90f0`)
- **Files:** `src/llama-context.h`, `src/llama-context.cpp`, `src/llama-graph.h`, `src/llama-graph.cpp`, `src/models/dspark.cpp`.
- **Problem:** On every Markov chain step in Inkling-2 DSpark (`block` steps per draft), `build_inkling2_head` multiplied `dspark_hidden_out` `[rank=256, n_embd=2304]` then projected through the full `target.output` `[2304, n_vocab_padded]` (expensive LM-head matmul at every draft step), plus ran the `pool_1d`/`step`/`tie_rank` argmax workaround and redundant unpadded vocab `ggml_cont` copies.
- **Fix:**
  - `llama_context::dspark_w2_eff_init()`: precomputes `dspark_w2_eff = target.output[:n_vocab_unpadded] @ dspark_hidden_out` once at context init in `F16 [rank, n_vocab_unpadded]`, chunked over `32,768` vocab rows to bound peak temp memory, disabled automatically if LoRAs, output scales, Hadamard rotations, or multi-GPU tensor split are active.
  - Folds `target.output @ dspark_hidden_out_b` into `base` (`base_inp = cur + dspark_hidden_out_b`) and slices `target.output` to `n_vocab_unpadded` before matmul.
  - Replaces the `pool_1d`/`step`/`tie_rank` workaround with direct `ggml_argmax(ctx0, valid)` (enabled by PR 5 `a77861652`), skipping the last chain step's unused `prev` argmax.
  - Avoids extra `ggml_cont` view when `base`/`logits`/`col` already match `n_vocab_unpadded`.
- **PR prep note:** Depends on PR 5 (`a77861652`) for deterministic first-index `ggml_argmax` tie-breaking. Benchmark DSpark draft latency (`dur(b,g,a)` / `tg` t/s) and verify zero acceptance-rate regression against fallback path before submitting.

### 7. PR 7 Candidate: Multi-Stream Banded Reserve for Non-Unified KV (`bc05d21ce`)
- **Files:** `src/llama-kv-cache.cpp`, `src/models/inkling2.cpp`.
- **Problem:** Under unified KV (`-kvu`), concurrent writing + prefill caused the generating slot to append 1-4 tokens onto the end of the prefilling slot's cells, shredding its full-attention KV cache into fragments and kicking all 16 full-attention layers back to dense naive attention (`592 -> 180 t/s`, reallocating every batch). Switching to non-unified per-slot streams (`--no-kv-unified -np 3 -c 600000`) eliminates cross-slot shredding completely, but `sched_reserve()` previously only reserved the banded path when `kv->get_n_stream() == 1` and `mask->ne[3] == 1`, inflating non-unified `3x200K` compute reserve to **`20,413 MiB`**.
- **Fix:**
  - Relax `llama_kv_cache_context::get_kv_banded_span()` reserve branch to work for any `n_stream` (`kv != nullptr && lctx == nullptr`).
  - Reshape `q_fa`, `rel_b`, and `sinks` by `n_q = n_tokens / n_stream` and `n_stream` in `inkling2.cpp` (allowing `n_stream > 1` during reserve; runtime banded still stays `n_stream == 1` per ubatch).
- **Impact:** Non-unified `3x200K` (`-b 1024 -ub 1024`) MTL0 compute reserve dropped from `20,413 MiB` -> **`3,257 MiB`** (`4,142 MiB` at `-b 2048 -ub 2048`), keeping concurrent prefill locked at `578-588 t/s` with `0` changed-size reallocs.

### 8. Local Server Uplift: Dynamic Decode-Priority Chunked Prefill (`721e555b8`)
- **Files:** `tools/server/server-context.cpp`.
- **Problem:** On a single GPU, large `-b 2048 -ub 2048` gives top solo prefill speed (`614 t/s`), but makes a concurrent writing slot pause ~1.7-3.4 s between token bursts while another slot prefills. Lowering `-b` statically slows down solo prefill.
- **Fix:** Track `decode_prefill_tick` in `server_context_impl`. While any slot is actively generating (`batch.size() > 0`), run 2 pure decode ticks per 1 mixed tick capped to `128` splittable prompt tokens (stays above `n_tokens_banded_min = 32`). When no slot is generating, reset and run full `-b 2048 -ub 2048`.
- **Verified numbers (`-np 3 -c 600000 --no-kv-unified -b 2048 -ub 2048`):**
  - Solo 12,026-token prefill: **`614.0 t/s`**, generation **`32.5-37.5 t/s`** (with `dspark_w2_eff`).
  - Concurrent write (160 tok) + 9,026-token prefill: writing slot stayed at **`11.2 t/s`** (`tg_3s = 10.9-11.8 t/s`, up from `0.08 t/s` unified stall and `5.26 t/s` static `1024`), background prefill averaged **`385.4 t/s`**, `0` changed-size graph reallocs (`36 / 0`).
- **PR note:** Currently hardcoded (`128` tokens, 2:1 decode-to-prefill ratio); if upstreaming to `llama-server`, expose as CLI flags (e.g. `--prefill-chunk-while-gen` / decode-priority ratio).

---

## Current Live Server Configuration

```sh
/Users/tomskeggs/dev/llama.cpp-inkling/build/bin/llama-server \
  -lv 4 \
  -m /Volumes/skeggsdrive/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf \
  --alias inkling -ngl 99 -fa on --jinja \
  --host 127.0.0.1 --port 8080 --log-timestamps \
  -np 3 -c 600000 --no-kv-unified \
  -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
  -b 2048 -ub 2048 --spec-draft-n-ubatch 64 -ctxcp 5 \
  --mmproj /Volumes/skeggsdrive/models/unsloth/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  -md /Volumes/skeggsdrive/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-p-min 0
```

- Footprint on Apple M5 Pro (64 GiB unified, `iogpu.wired_limit_mb = 59392`): `39.9 / 59.4 GiB` GPU used (**`19.1 GiB` headroom**), 3 isolated `200K` slots + 8 GiB RAM prompt cache (`--cache-ram`) for extra idle conversations.
