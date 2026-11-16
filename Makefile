GENERATOR ?= Ninja
DEBUG_DIR ?= build-debug
RELEASE_DIR ?= build-release
JOBS ?= $(shell sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

.PHONY: test release

test:
	cmake -S . -B "$(DEBUG_DIR)" -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=Debug -DYNNPACK_BUILD_TESTS=ON
	cmake --build "$(DEBUG_DIR)" --parallel "$(JOBS)"
	ctest --test-dir "$(DEBUG_DIR)" --output-on-failure --no-tests=error --parallel "$(JOBS)"

release:
	cmake -S . -B "$(RELEASE_DIR)" -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=Release -DYNNPACK_BUILD_TESTS=OFF
	cmake --build "$(RELEASE_DIR)" --parallel "$(JOBS)"