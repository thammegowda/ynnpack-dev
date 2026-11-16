# ynnpack-dev

ynnpack-dev is a standalone development fork of
[YNNPACK](https://github.com/google/XNNPACK/tree/78d255d4d6455f0249cfdd229756ee90e476e317/ynnpack),
**tailor-made for [Kidi](https://github.com/thammegowda/kidi)** as its low-level
CPU/CUDA kernel and composite backend. Architectural, API, and kernel decisions
in this fork are driven directly by what makes Kidi faster, leaner, and simpler.

Unlike upstream YNNPACK, `ynnpack-dev` removes the `ynnpack/subgraph` graph
compiler and the [Slinky](https://github.com/dsharlet/slinky) loop-lowering
runtime. [Kidi](https://github.com/thammegowda/kidi) already owns the high-level
operator graph, tensor lifetime/arena management, step replay, and thread-pool
scheduling (`WorkerPool`), so keeping a second graph IR and Slinky
JIT/scheduler in YNNPACK added compile latency, dispatch overhead, and
scheduling complexity without helping inference speed. `ynnpack-dev` instead
exposes direct, stateless C++20 (`-fno-rtti`) kernels and composites that
Kidi's runtime invokes directly over its own worker pool.

It retains YNNPACK's generated CPU microkernels, `base/` utilities,
`composites/` (dense/quantized/blockwise linear, flash/quantized attention,
RMSNorm, rotary, gated activations, HNSW, elementwise/tensor ops), CUDA
execution backend, tests, and benchmarks, while omitting the XNNPACK
compatibility layer and Bazel-only metadata.

## Origin and credits

The initial source was extracted from the YNNPACK tree in the original
[google/XNNPACK](https://github.com/google/XNNPACK) repository at commit
[`4691de78a2ed13e74a07108b0a4d033efa00ad4a`](https://github.com/google/XNNPACK/commit/4691de78a2ed13e74a07108b0a4d033efa00ad4a)
on 2026-09-24. That YNNPACK tree is identical to the tree at upstream commit
[`81a0804a40c26a9cf97ebb1acaff0c516d37f5b8`](https://github.com/google/XNNPACK/tree/81a0804a40c26a9cf97ebb1acaff0c516d37f5b8/ynnpack).

Credit for the original design and implementation belongs to the
[XNNPACK and YNNPACK contributors](https://github.com/google/XNNPACK/graphs/contributors).
Existing source-file copyright notices are retained.

## Upstream review

Last reviewed upstream commit (2026-10-22):
[`78d255d4d6455f0249cfdd229756ee90e476e317`](https://github.com/google/XNNPACK/commit/78d255d4d6455f0249cfdd229756ee90e476e317).
This is a review watermark, not a claim that every upstream change is integrated.
The extraction commit above remains the historical origin.

[Changes since the last review](https://github.com/google/XNNPACK/compare/78d255d4d6455f0249cfdd229756ee90e476e317...master)
and [the porting ledger](UPSTREAM.md) are the starting points for the next review.
The latest YNNPACK change within this snapshot is
[`533bf3f5`](https://github.com/google/XNNPACK/commit/533bf3f5afd2ad80016bfd05b81caa558dee729c).

For a path-filtered review, clone upstream once, then fetch and list new changes:

```sh
git clone --filter=blob:none --no-checkout https://github.com/google/XNNPACK.git ../XNNPACK-upstream
git -C ../XNNPACK-upstream fetch origin master
git -C ../XNNPACK-upstream log --reverse --format='%h %cs %s' \
	78d255d4d6455f0249cfdd229756ee90e476e317..origin/master \
	-- ynnpack cmake/DownloadCpuinfo.cmake
```

Review the existing backlog as well as the new delta. After reviewing a snapshot,
record each relevant change's disposition in the ledger, then advance this
watermark, compare link, source link, and log command together. Unported items
stay in the ledger even when the watermark advances.

## Dependencies

Initialize the pinned dependencies after cloning:

```sh
git submodule update --init --recursive
```

| Dependency | Purpose | License |
| --- | --- | --- |
| [cpuinfo](https://github.com/pytorch/cpuinfo) | Runtime CPU feature detection | [BSD 2-Clause](third_party/cpuinfo/LICENSE) |
| [GoogleTest](https://github.com/google/googletest) | Unit tests | [BSD 3-Clause](third_party/googletest/LICENSE) |
| [Google Benchmark](https://github.com/google/benchmark) | Microbenchmarks | [Apache 2.0](third_party/benchmark/LICENSE) |

`cpuinfo` supports the default library build. GoogleTest and Google
Benchmark are added only when their corresponding build options are enabled.
An embedding project may provide a compatible `cpuinfo` target before adding
this project.

Python 3.10 or newer is required at build time to generate architecture-specific
kernels. The generators use only the Python standard library.

## Build

```sh
cmake -S . -B build
cmake --build build
```

To build and run the standalone test suite:

```sh
cmake -S . -B build-test -DYNNPACK_BUILD_TESTS=ON
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

To run just the composite suites (dense/quantized linear, attention,
blockwise/symmetric FFN, HNSW, elementwise/normalization):

```sh
ctest --test-dir build-test --output-on-failure -R '^ynnpack_composites_'
```

Add `-DYNNPACK_BUILD_BENCHMARKS=ON` to build the microbenchmarks.

## Embed with CMake

After initializing this repository's submodules, an application needs only:

```cmake
add_subdirectory(third_party/ynnpack-dev EXCLUDE_FROM_ALL)
target_link_libraries(my_app PRIVATE ynnpack ynnpack_composites)
```

`ynnpack` exports the public and source-root include paths and links `base/`
plus the `kernels/` libraries. Link `ynnpack_composites` as well when using its
direct C++20 operator composites. Python discovery and CPU feature detection are
managed by the library; no custom cpuinfo target or private include path is
needed on Apple Silicon.

Tests and benchmarks default to OFF. Native builds default to bundled cpuinfo;
Emscripten builds select Wasm kernels and default cpuinfo to OFF. ARM64 FP8,
FP8DOT4, SME, and SME2 default to ON only when compile checks accept the required
flags, headers, intrinsic types, and attributes. SVE additionally defaults to
OFF on Apple platforms; elsewhere its check also verifies SVE C++ type support.
SME and SME2 additionally default to OFF on Windows, where Clang's SME hazard
padding code generation is unsupported even when the intrinsic probes pass.
These checks do not execute target code, so they also work while cross-compiling.
Runtime dispatch still depends on CPU feature detection.

The `YNN_ENABLE_CPUINFO` and `YNN_ENABLE_ARM64_*` options remain available for
explicit overrides. When migrating an existing
build that forced these options, clear those cached values or use a fresh build
directory to pick up the new defaults.

## License

The extracted YNNPACK code remains under the BSD-style [LICENSE](LICENSE), which
credits Facebook, Inc. and its affiliates and Google LLC. Retain the copyright
notices, license conditions, and disclaimer when redistributing source or binaries.
Bundled dependencies retain their own licenses under `third_party/`.
