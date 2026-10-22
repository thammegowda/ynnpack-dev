# Upstream Porting Ledger

Review date: 2026-10-22. The reviewed-through commit and next-delta commands live
in [README.md](README.md#upstream-review). This is selective backporting, not a
complete synchronization. Commit IDs below refer to google/XNNPACK unless marked
local. Deferred entries remain actionable after the review watermark advances.

## Ported Correctness Fixes

| Commit | Change and local adaptation |
| --- | --- |
| `7a4a3adb` | Compare output ranks and extents before CSE merges. Added different-rank and different-extent regressions. |
| `663dae0d` | Crop every partial-reduction dimension even when broadcast dimensions disappear from the input. Added repeated parallel broadcast reduction. |
| `520e7d3c` | Guard removed-broadcast output extent assignment against a shorter extent vector. |
| `3755947f` | Do not rewrite transpose reductions carrying an initializer. Added initializer-preservation regression. |
| `52903dd7` | Require one contracted dimension for sum-to-dot and check dot definition status. |
| `bacf504c` | Preserve affine input parameters through sqrt folding and divide-to-rsqrt rewriting; update scalar/reference arithmetic and defaults together. |
| `43f07333` | Apply affine input parameters in generated fp32/fp64 sqrt kernels. Added non-default sqrt/rsqrt kernel cases. |
| `cfc41230` | Compare expm1/log/log1p/sin/cos/sqrt parameters in unary CSE equality and ordering. Adapted to existing switch defaults; added comparison regression. |
| `de9ce20d` | Correct packed binary/ternary input bounds and transposed int4 bounds. Added scalar-parameter and tensor-parameter transpose/dequantize regressions. |

Local prerequisite: the shared `ynnpack_kernels_unary_generated` dependency
ensures all generated unary includes exist before any architecture compiles.
This was independently present as a staged fix in the original fork checkout;
the worktree contains its own copy, without changing that checkout or its index.

## Already Covered Locally

| Commit | Disposition |
| --- | --- |
| `953ee73a` | AVX512 fp16 compile flags already use the appropriate AVX512 architecture library. |
| `0b12c6ac` | External raw-buffer descriptor isolation is independently implemented by local `f73fe64`. Reconcile descriptor details with later external-output fixes rather than duplicate it. |

## Deferred Correctness Work

| Commits | Next investigation |
| --- | --- |
| `e3549f40`, `e28d46bd` | External-output types, shapes, fusion, and descriptors; reconcile with local descriptor isolation. |
| `1c4953ef`, `be941925` | Extent-one broadcasts and stencil bounds. Do not import `d297c798` alone: `be941925` partially rolls it back. |
| `a3227540` | Low-rank transpose_a bounds. |
| `4b8dbe0e` | Zero-extent handling. |
| `4ac56de4` | Unpacked dot B bounds. |
| 2026-10-08/09 API-validation batch | Broadcast/broadcast_like, gather, concatenate/stack, transpose, stencil overflow, static_pad, dynamic-quantization output types, and iota. |
| `512faee0` | Coordinated NaN propagation across API, SIMD, and reductions; not a standalone one-line port. |

## Deferred Performance and Build Work

| Commits | Scope and dependencies |
| --- | --- |
| `ac0a5bb0`, `f6514b13` | AVX-VNNI uint8 x int2/int4/int8 kernels plus the Windows intrinsic fix. Benchmark as a pair. |
| `087cc738`, `533bf3f5` | Learned dot cost models and consistent, cache-aware scheduling. Coordinate with Slinky upgrades and benchmark representative Kidi graphs. |
| `0f1a69e6`, `63bb37fb` | Android task-count tuning and single-iteration serial loops. |
| `ffb554c7`, `2da55926`, `6c880557` | transpose_a scheduling, slice stride aliasing, and force_root workaround removal. |
| `83872075`, `44dda8ae`, `4a5fe376` | Quantized dot load and shape tuning. |
| `769fa518`, `eeffad00`, `e6e692f5` | Low-bit transpose and interleave kernels. |
| `c3c545fb` | Reconcile ARM i8mm int8 x int2 CMake disablement with generator/registration state. Still enabled locally. |
| `6b90b2f7` | Optional FP64 build support. |
| `521ed240`, `fdd49a8b`, `61a7a7d0`, `601349ae` | log/log1p accuracy improvements. |
| `70873bd5` | Constant transpose folding. |
| `ebd43ff3`, `bd5a5669` | ARM BF16/FP8/dequant conversion tuning; evaluate CPU-dependent regressions. |

Dependency upgrades are intentionally deferred: Slinky from
`97c346749bf0f7fbdc2b0b8e6ff0c380806a32e2` to
`a34d74bd9a3c6c0a4f5dec6a7430273ec9bb7f66`, and cpuinfo from
`8ce83db858065145192c97af90cb668ad72a12e9` to
`66ee79c038d70dad9f08705b2c9b3e58f6d8f512`.

XNNPACK shim-only and Bazel visibility changes are outside this fork's current
scope. SME/Oryon V3 and Hexagon work depends on actual target support.

## Validation of This Batch

Linux x86_64, GNU 13.3, Release, existing pinned dependencies:

- Full standalone build passed; 58 of 61 CTest executables passed, including dot
  when rerun with a 600-second limit (136 seconds; the initial 120-second limit
  was too short).
- The other three executables fail identically on the original fork: consistent
  sum-squared reductions, cross-ISA erf consistency (seed 78214), and random
  floating-point iota (seed 79532). These are not fixed in this batch.
- Focused shape-CSE, initialized reduction/transpose, repeated parallel reduction,
  affine sqrt fusion, packed int4 dequantization, and 570 sqrt/rsqrt kernel tests
  passed. Shape-CSE regressions were also confirmed failing before the fix.
- Ninja `missingdeps ynnpack_kernels_unary` found no missing generated dependencies.
- Android, ARM, Wasm, sanitizers, and Kidi integration were not validated.