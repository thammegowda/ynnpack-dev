# ynnpack-dev

ynnpack-dev is a standalone development fork of
[YNNPACK](https://github.com/google/XNNPACK/tree/4691de78a2ed13e74a07108b0a4d033efa00ad4a/ynnpack).
It retains YNNPACK's graph compiler, Slinky lowering, composites, tests, benchmarks,
and generated CPU kernels while omitting the XNNPACK compatibility layer and
Bazel-only metadata. Its API may evolve independently, including adding selected
XNNPACK compatibility where useful.

## Origin and credits

The initial source was extracted from the YNNPACK tree in the original
[google/XNNPACK](https://github.com/google/XNNPACK) repository at commit
[`4691de78a2ed13e74a07108b0a4d033efa00ad4a`](https://github.com/google/XNNPACK/commit/4691de78a2ed13e74a07108b0a4d033efa00ad4a)
on 2026-09-24. That YNNPACK tree is identical to the tree at upstream commit
[`81a0804a40c26a9cf97ebb1acaff0c516d37f5b8`](https://github.com/google/XNNPACK/tree/81a0804a40c26a9cf97ebb1acaff0c516d37f5b8/ynnpack).

Credit for the original design and implementation belongs to the
[XNNPACK and YNNPACK contributors](https://github.com/google/XNNPACK/graphs/contributors).
Existing source-file copyright notices are retained.

## Dependencies

Initialize the pinned dependencies after cloning:

```sh
git submodule update --init --recursive
```

| Dependency | Purpose | License |
| --- | --- | --- |
| [Slinky](https://github.com/dsharlet/slinky) | Loop lowering, scheduling, and execution | [MIT](third_party/slinky/LICENSE) |
| [cpuinfo](https://github.com/pytorch/cpuinfo) | Runtime CPU feature detection | [BSD 2-Clause](third_party/cpuinfo/LICENSE) |
| [GoogleTest](https://github.com/google/googletest) | Unit tests | [BSD 3-Clause](third_party/googletest/LICENSE) |
| [Google Benchmark](https://github.com/google/benchmark) | Microbenchmarks | [Apache 2.0](third_party/benchmark/LICENSE) |

Slinky and cpuinfo support the default library build. GoogleTest and Google
Benchmark are added only when their corresponding build options are enabled.
An embedding project may provide compatible `slinky_base` or `cpuinfo` targets
before adding this project; `YNNPACK_USE_SYSTEM_LIBS=ON` selects an installed
Slinky package.

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

Add `-DYNNPACK_BUILD_BENCHMARKS=ON` to build the microbenchmarks.

## License

The extracted YNNPACK code remains under the BSD-style [LICENSE](LICENSE), which
credits Facebook, Inc. and its affiliates and Google LLC. Retain the copyright
notices, license conditions, and disclaimer when redistributing source or binaries.
Bundled dependencies retain their own licenses under `third_party/`.
