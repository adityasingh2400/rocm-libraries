# Decoupled PrefetchGlobalRead (per-tensor PGR) — research, design and test plan

Status: **design proposal, no production code changed yet.**
Branch context: `users/andysu12/decouple_pgr`.

Goal: let tensor **A** and tensor **B** choose their prefetch/LDS-buffering
behaviour independently, instead of one global `PrefetchGlobalRead` (PGR) that
forces both operands into the same ping-pong scheme. The motivating (hero) case
is `F8F4 TN MT64x512x512` on gfx1250, which cannot be code-generated today
because A+B double-buffered exceeds the 320 KiB LDS budget.

Everything numeric in this document was produced by running the **real**
TensileLite solution derivation (`Solution.assignDerivedParameters`) on this
checkout — see [Reproducing the numbers](#8-reproducing-the-numbers).

---

## 1. Conclusion first

1. **The hero case is real, and it is marginal.** Coupled PGR=2 needs
   **329 728 B (322.00 KiB)** and is rejected with
   `Kernel Uses 329728 > 327680 bytes of LDS`. Making A single-buffered and B
   double-buffered needs **296 448 B (289.50 KiB)** and fits. With `TDMInst=3`
   (the real gfx1250 path) the numbers are 332 800 B → **299 520 B (292.50 KiB)**.
   Henry's estimate reaches the same conclusion but is conservative on the
   exact bytes (348 160 → 313 344), because it assumes a uniform `272/256`
   padding factor whereas the real derivation picks
   `LdsBlockSizePerPadA=1024, LdsPadA=16` and `LdsBlockSizePerPadB=4096,
   LdsPadB=32`. The practical consequence is worth flagging: **the coupled
   variant overflows by only 2 048 B**, so any change to auto-padding selection
   can flip this case either way. Tests must pin the derived footprints, not
   just the pass/fail verdict.

2. **The feature is a per-tensor generalisation of a configuration that already
   works.** Today `1LDSBuffer=1` gives *both* tensors one LDS block and
   `1LDSBuffer=0` gives *both* two blocks. Both endpoints are supported and
   tested. What is missing is the **mixed** point (A:1 block, B:2 blocks). This
   makes the change far less risky than "decouple the whole PGR pipeline".

3. **The mixed layout only fits if the per-buffer stride is *not* rounded up to a
   power of two**, so it must use the existing SGPR/VGPR base-swap path
   (`StoreSwapAddr`) rather than the inline XOR constant. This is not a new
   mechanism — the coupled hero case already selects `StoreSwapAddr`.

4. **The swap/reset emit sites are already per-tensor.** `localWriteSwapOffsets`,
   `localWriteResetOffsets`, `localReadSwapOffsets`, `localReadResetOffsets` and
   `tdmSwapLdsOffset` all take `tP`/`tc` and already keep per-tensor swap
   registers (`SwapA/SwapB`, `LocalWriteSwapAddrA/B`, `LocalReadSwapAddrA/B`,
   `tdmLdsSwapMaskA/B`). They consult the *global* `1LDSBuffer` and a *shared*
   stride constant `LdsOffsetA_Blk`. Those two reads are the core of the change.

5. **The loop pipeline should stay coupled.** Loop-trip accounting
   (`unrollLoopEntryEndCounter`, `closeLoop` `endCounter`), the NGLL/NLL count
   (`for remainPgr in range(PGR-1, 0, -1)`) and the SIA scheduler all assume one
   software-pipeline depth. Splitting *those* is a much larger project and is
   **not** required by the hero case. Recommendation: keep
   `PrefetchGlobalRead` as the single pipeline-depth knob and decouple only the
   LDS residency + swap state.

6. **Backward compatibility is achievable with zero kernel renames**, using the
   `LDSSegmentInterleave` naming idiom (only name the parameter when it is
   non-default). This matters: adding a new entry to `requiredParameters`
   unconditionally would rename every kernel and invalidate every tuned logic
   file.

---

## 2. The hero case, verified

`F8F4 TN MT64x512x512` = `MacDataTypeA=F8`, `MacDataTypeB=F4`, `DataType=F8`,
`TransposeA=True/TransposeB=False`, `MacroTile0=64`, `MacroTile1=512`,
`DepthU=512`, gfx1250 (`DeviceLDS = 327680`).

Derived per-tensor LDS footprints (auto `LdsPad`/`LdsBlockSizePerPad`):

| path | `LdsPadA`/`LBSPPA` | `LdsPadB`/`LBSPPB` | A aligned | B aligned |
| --- | --- | --- | --- | --- |
| non-TDM (`TDMInst=0`) | 16 / 1024 | 32 / 4096 | 33 280 B (32.50 KiB) | 131 584 B (128.50 KiB) |
| TDM (`TDMInst=3`) | 16 / 1024 | 32 / 1024 | 33 280 B (32.50 KiB) | 133 120 B (130.00 KiB) |

Totals against the 320 KiB cap:

| configuration | non-TDM | TDM | fits 320 KiB |
| --- | --- | --- | --- |
| `1LDSBuffer=1` (A:1 blk, B:1 blk) | 164 864 B (161.00 KiB) | 166 400 B (162.50 KiB) | yes |
| **coupled `PGR=2`** (A:2, B:2) | **329 728 B (322.00 KiB)** | **332 800 B (325.00 KiB)** | **no — rejected** |
| **decoupled A:1, B:2** | **296 448 B (289.50 KiB)** | **299 520 B (292.50 KiB)** | **yes** |
| decoupled A:2, B:1 | 198 144 B (193.50 KiB) | 199 680 B (195.00 KiB) | yes |

The coupled figure `329728` is exactly the number in the real rejection message,
which confirms the model:

```
reject: Kernel Uses 329728 > 327680 bytes of LDS
```

### Why single-buffer **A** rather than **B**

Single-buffering B would save far more LDS (B is ~4x A), so from a pure
byte-count view B is the obvious choice. Henry's choice of A is a *performance*
argument, and it is the right one:

* Barriers are workgroup-wide, so whichever tensor is single-buffered forces the
  same read → sync → write discipline for the iteration. The barrier **count**
  is therefore roughly the same for `1LDSBuffer=1`, mixed, and coupled `PGR=2`.
* What actually differs is the **issue distance** of each tensor's global/TDM
  load. A double-buffered tensor can issue its load for iteration *i+1* a full
  iteration early (into the opposite block); a single-buffered tensor cannot
  issue until its own reads of iteration *i* have drained.
* B is ~4x larger than A, so B's load latency dominates. Keeping the deep
  prefetch on B and paying the serialisation on the cheap A load is the
  latency-optimal split.

This also frames the value proposition precisely. Today's F8F4 gfx1250 configs
(e.g. `Tensile/Tests/common/gemm/gfx12/f8f4ss_gfx1250.yaml`) already work around
the LDS wall with `1LDSBuffer: [1]`, which gives up the ping-pong on **both**
operands. Decoupled PGR is the missing middle point of the tuning space:

```
1LDSBuffer=1            decoupled A1/B2            coupled PGR=2
min LDS, min overlap  →  mid LDS, B overlap kept  →  max LDS, max overlap
   161 KiB                   289.5 KiB                322 KiB (rejected)
```

Whether the middle point actually wins is exactly what the A0/B0 perf run in
[§7](#7-performance-validation-a0b0) must answer.

---

## 3. How PGR works today, and where it couples

### 3.1 Parameter surface

| location | role |
| --- | --- |
| `Tensile/Common/ValidParameters.py:337-384` | `"PrefetchGlobalRead": [0, 1, 2] + range(3, 17)` plus the doc comment for 0/1/2/≥3 |
| `Tensile/Common/GlobalParameters.py:482` | default `[1]` → `defaultSolution["PrefetchGlobalRead"] == 1` |
| `Tensile/Common/RequiredParameters.py:110` | in `requiredParameters`, so every kernel name carries `PGR<n>` |
| `Tensile/SolutionStructs/Naming.py:112` | abbreviation is algorithmic: uppercase letters only → `PrefetchGlobalRead` → `PGR` |

### 3.2 LDS budget — the single coupling point that matters

All in `Tensile/SolutionStructs/Solution.py`:

* `calcLdsNumBytesAB` (~3427) computes **per-tensor** raw/aligned bytes. Already
  per-tensor; no change needed. The `272/256`-style inflation in Henry's estimate
  is the `LdsBlockSizePerPad` term at line 3446.
* `numLdsBlk` (~4952-4963) collapses everything to **one** block count:

```python
numLdsBlk = 1 if state["1LDSBuffer"] == 1 else 2
if state["PrefetchGlobalRead"] >= 3:
  numLdsBlk = state["PrefetchGlobalRead"]
  state["1LDSBuffer"] = 0
if state["PrefetchGlobalRead"] >= 2 and state["DtlPlusLdsBuf"]:
  numLdsBlk = state["PrefetchGlobalRead"] + 1
```

* `setLdsOffsets` (~4965) lays out one block as `[A][MXSA][MXSB][Meta][B]` and
  replicates it with a single stride `offsetBlk`:

```python
state["LdsOffsetA_Blk"] = offsetBlk
...
ldsNumBytesAB = (numLdsBlk - 2) * offsetBlk + state["LdsOffsetB_Blk"] + ldsNumBytesB
```

* the `PrefetchGlobalRead` block (~5020-5071) computes `offsetBlk`, rounds it to
  a power of two when it can, decides `StoreSwapAddr`, and produces
  `ldsNumBytesAB`.
* the final gate (~5443-5448) rejects when `LdsNumBytes > MaxLDS`.
* `MaxLDS` defaults to `archCaps["DeviceLDS"]` (~1647): 65 536 (gfx90a/942),
  163 840 (gfx950), 327 680 (gfx1250).

**Consequence:** one `offsetBlk` and one `numLdsBlk` describe both operands. A
mixed configuration is unrepresentable in the current state, which is why the
work starts here.

### 3.3 Swap emit sites — already per-tensor

`Tensile/KernelWriterAssembly.py`:

| site | current gate | current stride |
| --- | --- | --- |
| `tdmSwapLdsOffset` (11860) | `not kernel["1LDSBuffer"]` | `kernel["LdsOffsetA_Blk"]` (11876, 11882-11884) |
| `localWriteSwapOffsets` (11891) | `False if kernel["1LDSBuffer"] else True` | `getSrc0Val` (11951): `sgpr("Swap%s"%tc)` / `vgpr("LocalWriteSwapAddr%s"%tc)` under `StoreSwapAddr`, else inline `hex(kernel["LdsOffsetA_Blk"])` |
| `localWriteResetOffsets` (12003) | `not kernel["1LDSBuffer"]` | `kernel["LdsOffsetA_Blk"]-1 \| ldsOOB` (12020) |
| `localReadSwapOffsets` (12999) | early-return on `kernel["1LDSBuffer"]` | `vgpr("LocalReadSwapAddr%s"%tc)` under `StoreSwapAddr`, else `hex(kernel["LdsOffsetA_Blk"])` (13057) |
| `localReadResetOffsets` (13075) | early-return on `kernel["1LDSBuffer"]` | `LocalReadSwapAddr%s` / `hex(LdsOffsetA_Blk-1)` (13128) |

Swap-register initialisation is likewise per-tensor but uses the shared stride:

```python
# KernelWriterAssembly.py:5868  (SGPR local-write path)
SAddU32(sgpr("Swap%s"%tc), sgpr("LocalWriteAddr%s"%tc), kernel["LdsOffsetA_Blk"])
SXorB32(sgpr("Swap%s"%tc), sgpr("Swap%s"%tc), sgpr("LocalWriteAddr%s"%tc))
```

That is `Swap{tc} = (addr{tc} + stride) ^ addr{tc}` — a per-lane two-way XOR
toggle between `addr` and `addr + stride`. **Making `stride` per tensor is
sufficient to give A and B independent block strides**, and this form does not
require the stride to be a power of two. The same pattern appears at 5871
(VGPR), 6142 (`lraSwapAddressesForDTLPad`), and in the subtile TDM path
(`Tensile/Components/Subtile/SubtileGREmit.py:979`, masks initialised ~1137).

Orchestration in `Tensile/KernelWriter.py` (`_loopBody` ~4770-4841,
`noLoadLoopBody` ~3886-3980, `kernelBody` ~5518-5554, `closeLoop` ~8108-8156)
already calls each of these once per tensor; the *timing* gate
(`isSwapAndResetLwoIter` / `isSwapLroIter`) is global and can stay global — a
single-buffered tensor simply emits nothing at that point.

### 3.4 Barrier and memory-token policy — needs per-tensor awareness

```python
# KernelWriter.py:1025  (SIA0 iteration assembly)
if kernel["1LDSBuffer"]:
  if localWriteCode.itemsSize() > 0:
    barrier.addComment0("1 LDS buffer: read-sync-write")
    barrier.add(SWaitCnt(dscnt=0)); barrier.add(SBarrier())
```

```python
# KernelWriter.py:7332
self.states.oneBufferScheduling = (kernel["1LDSBuffer"]) or \
    ((kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and
     self.states.numLDSBlk == kernel["PrefetchGlobalRead"])

# KernelWriter.py:7346  — rocisa memory tokens
if kernel["1LDSBuffer"]:
  memTokenLdsBuffer0 = memTokenLdsBuffer1 = 0
  memTokenLdsSplit  = [[1, 2], [1, 2]]     # same token both parities -> aliasing
else:
  memTokenLdsBuffer0, memTokenLdsBuffer1 = 0, 1
  memTokenLdsSplit  = [[0, 2], [1, 6]]     # distinct tokens -> overlap allowed
```

`memTokenLdsSplit` is indexed `[bufferParity][half]`, and `half` already
separates the A-side from the B-side (`Tensile/Component.py:234-239`,
`KernelWriterAssembly.py:11440-11584`). So the token space already has
per-tensor granularity; what is global is the **parity** state
(`ldsWriteTokenIdx`, `ldsReadTokenIdx`, `ldsTensorTokenIdx`, flipped at
`KernelWriter.py:3752/3923/4228/4267/4568/4815` and
`KernelWriterAssembly.py:8158`).

Because `s_barrier` is workgroup-wide, the safe policy for the mixed case is:
**emit the read-sync-write barrier when *either* tensor is single-buffered**
(i.e. `oneBufferScheduling = OneLdsBufferA or OneLdsBufferB`), while keeping
the parity flip **per tensor** so the double-buffered operand still gets
distinct tokens and can overlap.

### 3.5 What must stay coupled (explicitly out of scope)

These are all keyed to one pipeline depth and are **not** touched by this
proposal:

| site | why it stays |
| --- | --- |
| `KernelWriterAssembly.py:7761` `unrollLoopEntryEndCounter` | one loop-trip threshold table for PGR 0/1/2/≥3 |
| `KernelWriterAssembly.py:8006-8031` `closeLoop` `endCounter` | main loop runs PGR extra trips |
| `KernelWriter.py:5991-6008` NGLL loop | `for remainPgr in range(PGR-1, 0, -1)` |
| `KernelWriter.py:6014-6058` ord-NLL count | `NLLnum` from PGR/DTV |
| `Tensile/Components/SIA.py` (469-954) | GR/LW pairing assumes one prefetch depth |
| `KernelWriterAssembly.py:6590-6656` stagger removal | `pf = f(PGR)` |

If per-tensor *pipeline* depth is wanted later, that is a separate project on
top of this one.

---

## 4. Design

### 4.1 The one decision to confirm: parameter semantics

Henry's shorthand "PGRA1 + PGRB2" arithmetically means **A gets 1 LDS block, B
gets 2**. That does *not* line up with today's global `PrefetchGlobalRead=1`,
which allocates **2** blocks (`numLdsBlk = 1 if 1LDSBuffer==1 else 2`). So
"per-tensor PGR level" and "per-tensor LDS block count" are not the same scale,
and the parameter must pick one. Two candidate spellings:

**Option A — `PrefetchGlobalReadA` / `PrefetchGlobalReadB`** (matches Henry's
naming). Tri-state-plus: `[-1, 0, 1, 2]`, default `-1` = inherit. Per-tensor
value *is* the LDS block count for that tensor. Requires an explicit
compatibility rule so legacy `PGR=1` keeps 2 blocks:

```python
if state["PrefetchGlobalReadA"] == -1 and state["PrefetchGlobalReadB"] == -1:
    # legacy path — bit-identical to today
    numLdsBlkA = numLdsBlkB = legacy_numLdsBlk(PGR, 1LDSBuffer, DtlPlusLdsBuf)
else:
    # decoupled path
    numLdsBlk{tc} = max(1, PGR{tc} if PGR{tc} != -1 else PGR)
```

Cost: `PGRA=1` and legacy `PGR=1` mean different LDS block counts. That wart has
to be documented, and reviewers will trip over it.

**Option B — `1LDSBufferA` / `1LDSBufferB`** (generalise the knob that already
means exactly this). Tri-state `[-1, 0, 1]`, same semantics as `1LDSBuffer`,
resolved per tensor. Hero case becomes
`PrefetchGlobalRead: 2, 1LDSBufferA: 1, 1LDSBufferB: 0`. No semantic collision,
smallest validation surface, and `1LDSBuffer` becomes the "set both" shorthand.
Cost: the feature is not spelled "PGR", so it reads less like Henry's note.

**Recommendation: build Option B's *mechanism*, expose Option A's *spelling* as
a thin front end.** The mechanism (per-tensor block count, per-tensor stride,
per-tensor swap and token state) is ~90% of the work and is identical either
way, so the spelling can be settled late and cheaply:

```python
# front end, evaluated before the LDS budget
state["OneLdsBufferA"], state["OneLdsBufferB"] = resolvePerTensorBuffering(state)
```

Everything downstream reads only `OneLdsBuffer{A,B}` / `NumLdsBlk{A,B}` /
`LdsBlkStride{A,B}`. Concretely I suggest landing Phase 0-2 against the derived
keys, then adding `PrefetchGlobalReadA/B` in Phase 1 as the user-facing name
once you and Henry agree on the table above.

### 4.2 Kernel naming — do not rename existing kernels

`requiredParameters` feeds every kernel name, so adding an entry unconditionally
renames all kernels and invalidates all tuned logic. Use the existing
conditional-naming idiom from `LDSSegmentInterleave`
(`Tensile/SolutionStructs/Naming.py:223-226`):

```python
# Only name per-tensor PGR when it actually differs from the coupled setting, so a
# decoupled kernel is distinct from its baseline twin without tagging every other kernel.
if state.get("NumLdsBlkA") != state.get("NumLdsBlkB"):
    requiredParametersTemp.add("PrefetchGlobalReadA")
    requiredParametersTemp.add("PrefetchGlobalReadB")
```

Verification for this is cheap and belongs in the test plan: assert that a
coupled solution's `getSolutionNameFull` is byte-identical before/after the
change.

### 4.3 LDS layout for the mixed case

Today, two blocks with a single stride:

```
[ A | MXSA | MXSB | Meta | B ]  [ A | MXSA | MXSB | Meta | B ]
0                               offsetBlk (power of two when it fits)
```

For A:1 block, B:2 blocks the natural layout is "single-buffered tensors once,
double-buffered tensors replicated":

```
[ A ][ B blk0 ][ B blk1 ]
0    33280     164864          (non-TDM hero numbers; stride = alignedB = 131584)
```

Stride for B is `alignedB`, which is **not** a power of two. Checked against the
non-TDM hero case (`a = 33 280`, `b = 131 584`, `MaxLDS = 327 680`):

| layout | swap form | stride | total | fits 320 KiB |
| --- | --- | --- | --- | --- |
| `[A][B0][B1]` | SGPR/VGPR base swap (`StoreSwapAddr`) | 131 584 | 296 448 B (289.50 KiB) | **yes** |
| `[B0][B1][A]` | inline XOR constant | 262 144 | 427 008 B (417.00 KiB) | no |
| `[A][B0][B1]`, B base aligned to the stride | inline XOR constant | 262 144 | 655 872 B (640.50 KiB) | no |

So: **a mixed configuration must force `StoreSwapAddr = True`.** The inline-XOR
fast path is only correct for a power-of-two stride (for `off < X`,
`off ^ X == off + X` only when `X` is a single bit), and rounding up to a power
of two defeats the entire purpose of the feature. The base-swap path already
exists and the coupled hero case already selects it, so this is a constraint to
assert, not new code to write.

### 4.4 Derived state to introduce

| key | meaning | legacy value |
| --- | --- | --- |
| `OneLdsBufferA/B` | resolved per-tensor single-buffer flag | `1LDSBuffer` |
| `NumLdsBlkA/B` | per-tensor LDS block count | `NumLdsBlk` |
| `LdsBlkStrideA/B` | per-tensor block stride (bytes) | `LdsOffsetA_Blk` |
| `LdsBlkBaseA/B` | per-tensor block-0 base | `LdsOffsetA` / `LdsOffsetB` |

Keeping `NumLdsBlk` / `LdsOffsetA_Blk` as aliases of the coupled values means
every untouched consumer keeps working during Phase 0.

### 4.5 Rejections to add

A mixed configuration must be rejected together with any feature that assumes a
single shared stride or a single buffer parity:

| condition | reason |
| --- | --- |
| `ExpandPointerSwap` | tracks one swap offset in Python (`localWriteSwapByteOffset`) |
| `NumLdsBlk >= 3` / `IncLdsBufSwitch` / `PGR >= 3` | round-robin `LDSBufferWriteInc` is one shared SGPR |
| `useCommonSgprSwap` | one `SwapCommon` SGPR emitted at A only |
| `DtlPlusLdsBuf`, `DirectToLds` | PGR+1 block scheme is shared |
| `ProblemType["Sparse"]` | metadata swap is tied to A/B swap |
| `LDSSegmentInterleave == 1` | its own interleaved layout + `blockSpan` budget |
| `UseSubtileImpl` | separate codegen body; enable later in its own phase |
| `1LDSBuffer == 1` (both) | degenerate; use the coupled knob |
| not `StoreSwapAddr`-capable | see §4.3 |

Start maximally restrictive and relax per phase. Every rejection needs a
`printRejectionReason` string and a unit test.

---

## 5. Phased development plan

Each phase has an explicit **gate** that must be green before the next starts.
The gate for Phase 0 is the most important thing in this document: it is what
makes the refactor safe.

### Phase 0 — per-tensor plumbing, zero behaviour change

* Add the derived keys of §4.4, initialised to the coupled values.
* Replace, in the five emit sites of §3.3, `kernel["1LDSBuffer"]` with
  `kernel["OneLdsBuffer%s" % tc]` and stride reads of `kernel["LdsOffsetA_Blk"]`
  with `kernel["LdsBlkStride%s" % tc]`.
* Split the token parity into `ldsWriteTokenIdx{A,B}` etc., all flipped
  together for now.
* No new user-facing parameter, no new YAML.

**Gate: byte-identical assembly.** Run the whole codegen characterization suite;
zero snapshot diffs allowed. Also assert `getSolutionNameFull` is unchanged.
This phase is where a mistake is cheapest to find, so it should be a standalone
reviewable commit.

### Phase 1 — parameter surface + LDS budget

* `validParameters` + `defaultBenchmarkCommonParameters` + conditional naming
  (§4.2).
* `resolvePerTensorBuffering`, per-tensor `NumLdsBlk{A,B}`, mixed layout and
  budget (§4.3), forced `StoreSwapAddr`, rejections (§4.5).
* Host-side only; codegen still refuses mixed via the rejections.

**Gate:** new pure-logic pytest (§6.1) shows the hero case rejected at
322.00 KiB coupled and accepted at 289.50 KiB mixed; characterization snapshots
still unchanged for every legacy config.

### Phase 2 — codegen for the mixed case, non-TDM

* Per-tensor swap-register init using `LdsBlkStride{tc}` (5868/5871/6142).
* `oneBufferScheduling = OneLdsBufferA or OneLdsBufferB`; barrier at
  `KernelWriter.py:1025` and `1162` driven by the same.
* Per-tensor token parity flip.
* Drop the mixed rejection for the plain buffer-load path.

**Gate:** emit + assemble the hero kernel; new codegen characterization snapshot
(§6.2); manual read of the swap/barrier sequence in the emitted asm. This is the
phase where a human should actually read the generated assembly.

### Phase 3 — gfx1250 TDM path

* `tdmSwapLdsOffset` (11876, 11882-11884) per-tensor stride; note the
  `StoreSwapAddr` branch there uses a compare/select add rather than XOR and so
  needs the per-tensor stride in both arms.
* Subtile `globalReadLDSBufferSwap` / `tdmLdsSwapMask{tc}` init.
* Re-check the TDM barrier interplay: `KernelWriter.py:4767` (PGR1+TDM double
  barrier) and `4773` (PGR2+TDM+SIA0 cross-wave swap sync) both branch on the
  global PGR and on both TDM flags.

**Gate:** gfx1250 codegen characterization + first correctness run on hardware.

### Phase 4 — validation and tuning

* Correctness YAML tests (§6.3) on the dev machine and on A0/B0.
* Perf comparison (§7) against both `1LDSBuffer=1` and, where it fits, coupled
  `PGR=2`.
* Only then consider relaxing the §4.5 rejections (subtile, sparse, DTL).

---

## 6. Test plan

Three tiers, cheapest first. Tier 1 and 2 need **no GPU** — they run on any host
with the ROCm toolchain and rocisa installed, which is what makes the inner dev
loop fast.

### 6.1 Tier 1 — pure-logic pytest (no GPU)

Model on `Tensile/Tests/unit/test_segment_interleave.py`, which tests the
`segment_interleave` oracle with a `_FakeDataType` and imports nothing from
rocisa. Same shape here: put the per-tensor budget maths in a small pure module
(`Tensile/SolutionStructs/pgr_buffering.py`) so it can be unit-tested in
milliseconds, then have `Solution.py` call it — exactly the
`segment_interleave.py` / `Solution.py` relationship that already exists.

Gotcha worth knowing before you set up the loop: keeping the *module* rocisa-free
is what makes these tests fast and robust, but it does **not** let you run them
without rocisa installed. `Tensile/Tests/unit/conftest.py` imports
`streamk5_test_helpers` → `KernelWriterAssembly` → `rocisa`, so collection fails
for the whole directory if rocisa is missing. Install it once
(`invoke rocisa`); after that the suite itself runs in ~0.1 s.

New file `Tensile/Tests/unit/test_pgr_decouple_budget.py`:

| test | asserts |
| --- | --- |
| `test_hero_coupled_rejected` | A=33280, B=131584, coupled → 329728 > 327680 |
| `test_hero_mixed_a1b2_fits` | mixed A:1/B:2 → 296448 ≤ 327680 |
| `test_hero_mixed_tdm_fits` | TDM sizes (33280/133120) → 299520 ≤ 327680 |
| `test_mixed_forces_store_swap_addr` | pow2-stride layouts exceed MaxLDS, so mixed ⇒ `StoreSwapAddr` |
| `test_legacy_inherit_is_bitwise_identical` | both `-1` reproduces `numLdsBlk` for PGR 0/1/2/3..16 × `1LDSBuffer` 0/1 × `DtlPlusLdsBuf` 0/1 |
| `test_mixed_rejects_<feature>` | one per row of §4.5 |
| `test_kernel_name_unchanged_when_coupled` | `getSolutionNameFull` byte-identical for a coupled state |
| `test_kernel_name_tagged_when_mixed` | mixed state name differs from coupled twin |

### 6.2 Tier 2 — codegen characterization (no GPU, needs rocisa + amdclang++)

The suite under `Tensile/Tests/unit/characterization/_codegen/` drives
`config YAML → Solution → KernelWriterAssembly → assembly text` on the host and
snapshots the canonicalised assembly (syrupy). Verified working on this
checkout: `test_emit_gfx1250_char.py` passes in ~36 s.

Two uses:

1. **Regression net for Phase 0/1.** Run the existing suites unchanged; any
   snapshot diff is a bug. This is the highest-value test in the whole plan.
2. **New designed configs** under
   `characterization/_codegen/data/test_data/_designed/gfx1250/`, consumed by a
   new `test_r*_pgr_decouple_char.py` that follows the existing
   `emit_kernels_from_config` pattern. Derive them from the probe configs staged
   in `configs/` next to this document:
   * `hero_f8f4_gfx1250.yaml` — hero case, non-TDM
   * `hero_f8f4_gfx1250_tdm.yaml` — hero case with `TDMInst=3`
   both of which already run against the *current* tree (they fork
   `1LDSBuffer: [0, 1]` to show the rejected coupled variant next to the
   per-tensor footprints). Once the parameter exists, swap the fork for the
   decoupled pair.

Note the corpus gate: `Tensile/Tests/unit/test_input_yaml_corpus_clean.py` type-
checks every YAML under `Tensile/Tests/common/` and
`Tensile/Tests/unit/test_data/` against `validParameters`. **Do not move the
staged configs into those trees until the parameter exists**, or CI goes red.
That is why they are parked under `docs/decouple-pgr/configs/` for now.

### 6.3 Tier 3 — tensilelite test YAML (needs GPU)

`Tensile/Tests/common/**/*.yaml` configs are collected by
`Tensile/Tests/common/test_config.py`, which builds the kernels, benchmarks, and
validates against a CPU reference. `NumElementsToValidate: -1` validates every
element — that is the correctness signal.

Planned additions:

| file | arch | content |
| --- | --- | --- |
| `common/gemm/gfx12/pgr_decouple_gfx1250.yaml` | gfx1250 | hero case + TDM/non-TDM × A1B2 / A2B1 / coupled / inherit — **draft staged at `configs/pgr_decouple_gfx1250.yaml`** |
| `common/gemm/gfx950/pgr_decouple_gfx950.yaml` | gfx950 | scaled analog that fits 160 KiB (see below) |

Both must carry the `TestParameters.marks` skip list for every other arch, the
way `f8f4ss_gfx1250.yaml:2` does, and `PrintSolutionRejectionReason: True` so a
rejection is visible instead of silently producing zero solutions.

Coverage matrix to fork over, kept small enough to stay a CI test:

* `PrefetchGlobalReadA/B` (or `1LDSBufferA/B`): `(1,2)`, `(2,1)`, `(2,2)`, `(-1,-1)`
* `TDMInst`: `[0, 3]` on gfx1250
* `ScheduleIterAlg`: `[0, 3]` — SIA0 and SIA3 take different barrier paths
  (`KernelWriter.py:1025` vs the SIA3 scheduler)
* `PrefetchLocalRead`: `[1]` initially, widen after Phase 4
* problem sizes: one K-tail-free size and one with a tail, since the tail loop has
  its own reset path (`localWriteResetOffsets`, `localReadResetOffsets`)

**Picking a gfx950 dev case.** For an arch with `MaxLDS = M`, a config exercises
the feature iff coupled overflows and mixed fits:

```
2*(a + b) > M   and   a + 2b <= M      (single-buffer A)
```

with `a`, `b` the aligned per-tensor footprints. For gfx950 (`M = 163840`) a
worked candidate is F8/F4 `MT0=32, MT1=256, DepthU=512`: `a ≈ 16640`,
`b ≈ 65792` → coupled `164864 > 163840` (rejected by 1 KiB), mixed
`148224 ≤ 163840` (fits). Confirm with the probe tool before committing the
YAML — auto `LdsPad`/`LdsBlockSizePerPad` selection differs per arch and per
`TDMInst`, as the non-TDM vs TDM columns in §2 show.

### 6.4 What each tier does *not* cover

* Tier 1/2 cannot catch a race. A wrong barrier or a wrong memory token produces
  perfectly valid-looking assembly. Only Tier 3 with full validation on real
  hardware, ideally with more than one wave and a K-loop long enough to cycle
  the buffers many times, will catch it.
* Snapshot tests pin *current* behaviour, not *correct* behaviour. A Phase-2
  snapshot must be reviewed by a human before it is accepted as golden.

---

## 7. Performance validation (A0/B0)

Once Phase 3 is functionally correct, the question is whether the middle point
of §2 is worth having. Suggested comparison, hero shape and neighbours:

| variant | expectation |
| --- | --- |
| `1LDSBuffer=1` (today's shipping workaround) | baseline |
| decoupled A:1, B:2 | should win — B keeps its deep prefetch |
| decoupled A:2, B:1 | control; should be *worse* than A:1/B:2 despite using less LDS, which is the cleanest confirmation that issue distance and not LDS bytes is what matters |
| coupled `PGR=2` at a shape where it fits (e.g. `DepthU=256`) | upper bound |

Also worth recording, because they are the reasons the middle point could fail
to pay off: achieved occupancy (LDS per workgroup went from 161 to 289.5 KiB,
which can cost a workgroup per CU), barrier count per K iteration, and
`s_wait_tensorcnt` stalls on the TDM path.

---

## 8. Reproducing the numbers

`lds_budget_probe.py` next to this document drives the real derivation through
the CPU-only characterization harness and prints the per-tensor LDS state.

```bash
cd projects/hipblaslt/tensilelite
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH

# per-tensor footprints + budget for a config, at a chosen arch
python3 docs/decouple-pgr/lds_budget_probe.py \
    docs/decouple-pgr/configs/hero_f8f4_gfx1250.yaml gfx1250

# what-if table: coupled vs mixed vs 1LDSBuffer for given a/b/MaxLDS
python3 docs/decouple-pgr/lds_budget_probe.py --budget 33280 131584 327680
```

The rejected coupled variant needs `NoReject: [True]` in `ForkParameters` to
survive derivation far enough to dump its state; otherwise the solution is
dropped and the probe prints `0 solution(s)` with the reason on stderr.

### Environment used

* `rocisa` builds host-only from the ROCm toolchain — no GPU needed. On a bare
  Ubuntu 24.04 host: `rocm-llvm hip-dev hipcc7.2.4 rocm-cmake python3-dev`, then
  `pip install ./rocisa`. Watch out for Ubuntu's own `hipcc` 5.7.1 shadowing the
  ROCm one; `find_package(hip)` needs `/opt/rocm/bin/hipcc` to exist.
* Tier 1 + Tier 2 both verified on that setup
  (`test_segment_interleave.py` 31 passed; `test_emit_gfx1250_char.py`
  1 snapshot passed).

---

## 9. Dev loop

### On the dev machine (190 FFM), repo at `/home/andysu/data3/andysu12/public/rocm-libraries`

```bash
cd /home/andysu/data3/andysu12/public/rocm-libraries/projects/hipblaslt/tensilelite

# once, or after rocisa C++/cmake changes
invoke rocisa

# Tier 1 — milliseconds, run on every edit
pytest Tensile/Tests/unit/test_pgr_decouple_budget.py -q

# Tier 2, regression net — the Phase 0/1 gate; zero snapshot diffs allowed
pytest Tensile/Tests/unit/characterization/_codegen -q --snapshot-warn-unused

# Tier 2, new designed configs for the decoupled kernels
pytest Tensile/Tests/unit/characterization/_codegen/test_r7_pgr_decouple_char.py -q

# Tier 3 — needs a GPU; scope to the new config
invoke build-client
pytest Tensile/Tests/common -q -k pgr_decouple
```

`tox -e unit` also works and installs rocisa itself; `tox -e py3` additionally
runs `build-client` (the slow part).

### Cross-machine split (dev machine builds, A0/B0 runs)

`Tensile/Tests/conftest.py` already supports splitting the build from the run,
which is the right shape for "generate on 190, measure on A0/B0":

```bash
# on the dev machine
pytest Tensile/Tests/common -k pgr_decouple --build-only --artifact-dir /shared/pgr-artifacts

# on A0/B0
pytest Tensile/Tests/common -k pgr_decouple --use-cache  --artifact-dir /shared/pgr-artifacts
```

Use `--gpu-targets gfx1250` to override hardware auto-detection when generating
on a machine that is not the target.

### Commit shape

One commit per phase, each independently reviewable and each with its gate
result in the message. Phase 0 in particular should be reviewable as "pure
refactor, zero snapshot diff".
