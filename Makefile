BINARY = blocknet-vanity

CUDA ?= auto

# Use native CPU features on macOS (Apple Silicon instruction scheduling)
ifeq ($(shell uname -s),Darwin)
export RUSTFLAGS += -C target-cpu=native
endif

ifeq ($(CUDA),1)
all: build-cuda
else ifeq ($(CUDA),0)
all: build
else
all: build-auto
endif

build:
	cargo build --release
	cp target/release/$(BINARY) .

build-auto:
	@if [ "$$(uname -s)" = "Linux" ] && command -v nvcc >/dev/null 2>&1; then \
		echo "Auto-detected CUDA toolchain (Linux + nvcc). Building with --features cuda."; \
		cargo build --release --features cuda; \
	else \
		echo "CUDA toolchain not detected. Building CPU release."; \
		cargo build --release; \
	fi
	cp target/release/$(BINARY) .

build-cuda:
	@if [ "$$(uname -s)" != "Linux" ]; then \
		echo "CUDA build is Linux-only. Run this on your Arch + 4070 Ti machine."; \
		exit 1; \
	fi
	cargo build --release --features cuda
	cp target/release/$(BINARY) .

clean:
	cargo clean
	rm -f $(BINARY)

.PHONY: all build build-auto build-cuda clean
