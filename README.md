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

YNNPACK uses [Slinky](https://github.com/dsharlet/slinky) for loop lowering,
scheduling, and execution. An embedding build may define the Slinky CMake targets
before adding this directory. A standalone build can provide `SLINKY_SOURCE_DIR`,
or allow YNNPACK's pinned Slinky download.

Python 3.10 or newer is required at build time to generate architecture-specific
kernels. The generators use only the Python standard library.

## Build

```sh
cmake -S . -B build -DSLINKY_SOURCE_DIR=/path/to/slinky
cmake --build build
```

## License

The extracted YNNPACK code remains under the BSD-style [LICENSE](LICENSE), which
credits Facebook, Inc. and its affiliates and Google LLC. Retain the copyright
notices, license conditions, and disclaimer when redistributing source or binaries.
Slinky is separately licensed under the MIT License.
