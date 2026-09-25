# Native CI repair journal

## 2026-09-25: ARM failures

- Reproduced eight exact reduction-consistency failures on Apple Silicon.
  FP32/FP64 contiguous kernels used physical SIMD widths instead of the shared
  logical width, changing summation order. The common-width setting fixes all
  four FP32/FP64 cases without changing assertions.
- BF16 also accumulated through BFDOT, whose rounding differs from normal FP32
  addition. Keep BFDOT for per-input work, accumulate with FP32 addition, and use
  the same logical ordering as the scalar path. All 76 existing consistency
  cases and the independent accuracy tests pass. Added a deterministic BF16
  halfway-rounding regression for contiguous and strided reductions.
- Fusion aborted because the existing int8-to-uint8 sign-bit conversion was
  not registered for NEON. Registered it; the formerly aborting fusion case and
  all 40 generated conversion checks pass.
- Two fusion tests incorrectly required a uint8-dot preference on every CPU.
  They now check the signed no-rewrite state as well as the unsigned rewrite.
- The dequantize-dot test assumed one fixed packing layout. SME adds an input
  transpose. Check exact fused inputs, the removed accumulator, and the dead
  intermediate instead of architecture-specific node counts. Full fusion passes.

## Verification and remaining work

- Apple Silicon Debug: all 60 CTest executables pass, with no exclusions.
- Apple Silicon Release: reduction accuracy, exact consistency, unary, and
  fusion checks all pass (4/4). Post-format consistency and fusion reruns pass.
- Disassembly confirms that BF16 reductions retain BFDOT and use FP32 additions
  for accumulation. Generator syntax validation and diff whitespace checks pass.
- Earlier Windows RTTI, x64 dot ISA mapping, and Node24 workflow fixes remain
  in the same uncommitted change set. Hosted Windows/Linux verification requires
  the next push; no tests have been excluded.
- Advanced ISA build-coverage audit, including AVX512-FP16, remains separate
  unfinished work. Host CPU capabilities must not determine compiled variants.
- No commit or push performed.