# Session Synthesis: Defect Triage and Structural Remediation

**Date:** 2026-02-17
**Author:** Root session
**Provenance:** Synthesis across 6 subagent essays and 4 user-reported defects

---

## What Was Accomplished

This session started with a briefing about 4 urgent defects threatening to be
lost across session boundaries. Each defect received a different response:

### Defect 1: src/ resurrection (policy response)

**Problem:** Agents were using the old `src/futudiffu/` code instead of
building canonical implementations in `src_ii/`.

**Response:** CLAUDE.md updated with explicit "src/ Freeze Policy" section.
`src/futudiffu/` is frozen — no new imports, no modifications, no agents
touching it. All new work in `src_ii/` and `scripts_ii/`.

**Enforcement:** Pokayoke guard added to `scripts/pokayoke_inline_check.py`
(Check 3). AST-based import detection + regex sys.path detection. 65
existing violations grandfathered. New violations fail the check.

### Defect 2: ZMQ necromancy (architecture replacement)

**Problem:** The ZMQ server had a track record of async deadlocks, REQ socket
poisoning, and concurrency failures that agents misattributed to compilation
stalls. Agents kept resurrecting it.

**Response:** CLAUDE.md updated: "ZeroMQ is dead architecture." Full FastAPI
server implemented in src_ii/:

| File | Purpose | Status |
|------|---------|--------|
| `src_ii/server.py` | FastAPI app, all 20 RPCs, ModelBackend protocol | Done |
| `src_ii/server_models.py` | Pydantic request/response types | Done |
| `src_ii/http_client.py` | Drop-in HTTPInferenceClient (lazy torch) | Done |
| `scripts_ii/launch_server.py` | Argparse + uvicorn launcher | Done |
| `tests/test_fastapi_server.py` | 44 mock-backend tests | Done (needs deps) |

**Blocked by:** User adding fastapi/uvicorn/pydantic/httpx to pyproject.toml
and running uv sync from Windows PowerShell.

### Defect 3: Root Claude context drowning (behavioral correction)

**Problem:** Root Claude was diving into code reads, losing the thread of
what subagents were doing, and spending messages recovering awareness instead
of synthesizing results.

**Response:** This session operated under the orchestration principles:
- Root Claude read NO source files directly
- All code exploration delegated to subagents with reading lists and rubrics
- Subagents returned essays; root Claude wrote synthesis essays
- 6 subagent dispatches, all with reading lists + rubrics, no step-by-step
  micromanagement

### Defect 4: SageAttention integration refusal (kernel completion)

**Problem:** A subagent refused to implement SageAttention-compatible
FlexAttention block masks, citing an outdated design doc (Section 8 of
flexattention_batch_packing.md). The masked kernels already existed but
weren't wired in. No pushback from root Claude.

**Response:** Three-phase resolution:

1. **Audit** (`essay_sage_integration_audit.md`): Subagent traced the exact
   gap — two call sites construct a FlexAttention `BlockMask` (wrong type)
   instead of a `uint8` tensor (right type for Sage dispatch). The dispatch
   logic, API wrapper, and kernels are all ready.

2. **Synthesis** (`essay_synthesis_sage_integration_and_src_ii_gap.md`):
   Root Claude identified that the integration must happen in `src_ii/`
   (since `src/` is frozen), and that `src_ii/` needs an attention dispatch
   module built from scratch with unified masked-Sage as the default.

3. **Masked backward kernels** (`essay_masked_sage_backward.md`): The user
   correctly identified that "forward-only masked Sage" was being treated
   as a hard barrier when it's just a kernel task. Subagent implemented:
   - `_sage_attn_fwd_int8qk_bf16pv_masked_lse` (masked forward + LSE)
   - `_sage_attn_bwd_dkdv_int8_masked` (masked backward dK/dV)
   - `_sage_attn_bwd_dq_int8_masked` (masked backward dQ)
   - `sage_attn_masked_op` custom op with register_autograd + register_fake

   The quadrant is now complete: unmasked forward, unmasked backward,
   masked forward, masked backward — all with STE through INT8 quantization.

---

## What Remains

### Immediate (before Unit 4 can start)

1. **pyproject.toml deps + uv sync** — User must add fastapi/uvicorn/
   pydantic/httpx and run uv sync from Windows PowerShell.

2. **Run tests** — `pytest tests/test_fastapi_server.py` after deps install.

3. **`src_ii/attention.py`** — Unified attention dispatch module. Takes Q,
   K, V, optional uint8 block mask, backend config. Routes to masked Sage,
   unmasked Sage, or SDPA. No FlexAttention. No type-driven dispatch. One
   type, one path.

4. **`src_ii/block_mask.py`** — uint8 block mask construction from packing
   info. ~30 lines.

5. **`src_ii/forward_packed.py`** — Packed forward path using the above.

6. **Update Section 8** of `docs/flexattention_batch_packing.md` to reflect
   the current state (masked Sage exists, is wired, is the default).

### Then: Unit 4 (PINKIFY / THISNOTTHAT)

The goal workstream: implement PINKIFY (pinker = preferred) and THISNOTTHAT
(similarity to reference images) as BTRM heads, train them end-to-end,
verify persist/load bit-for-bit. This requires the attention dispatch and
packed forward path in src_ii/ to be functional.

---

## Pokayoke Metrics

| Metric | Value | Direction |
|--------|-------|-----------|
| src/ freeze exceptions | 65 | Should shrink monotonically |
| Inline reimplementation exceptions | 1 (false positive) | Stable |
| Kernel quadrant completion | 4/4 | Complete |
| FastAPI RPC parity | 20/20 + health | Complete |
| Mock backend test count | 44 | Should grow |

---

## Process Observations

1. **Subagent reading time dominates.** The kernel subagent spent ~60% of
   its tokens reading existing kernels before writing any code. This is
   correct — the reading list ensures the subagent understands existing
   conventions before adding to them. But it means kernel-heavy tasks
   consume significant context budget.

2. **Quota as forcing function.** Two subagents hit rate limits. The FastAPI
   server agent got most of its work done; the kernel agent on its first
   attempt got nothing written. The difference: the server agent wrote
   files early (models first, then server, then essay). The kernel agent
   read everything first. For quota-sensitive environments, a "write early,
   iterate" strategy is more robust than "read everything, then write."

3. **The "just a kernel" correction was correct.** Root Claude initially
   deferred the masked backward kernels as an architectural decision. The
   user correctly identified it as a mechanical pattern replication. The
   fix: don't categorize engineering tasks by their perceived difficulty.
   Categorize by their structure. "Add HAS_BLOCK_MASK to an existing
   kernel" is a pattern replication, regardless of whether the domain is
   Triton or argparse.
