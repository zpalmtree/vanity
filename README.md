# blocknet-vanity

Vanity address generator for Blocknet. Brute-forces keypairs to find addresses matching a given prefix and/or suffix.

## how it works

A Blocknet address is `base58(spend_public_key || view_public_key || checksum4)`, where `checksum4` is the first 4 bytes of `SHA3-256("blocknet_stealth_address_checksum" || "blocknet_mainnet" || spend_public_key || view_public_key)`. The generator fixes the view keypair per thread and increments the spend scalar, requiring only one scalar multiplication per candidate. Matching is case-insensitive.

## download

Pre-built binaries at [blocknetcrypto.com](https://blocknetcrypto.com). Single binary, no extra files needed.

## build from source

Requires Rust 1.75+.

### linux

```
sudo apt install build-essential
make
```

### macos

```
xcode-select --install
make
```

### windows (msys2)

Install MSYS2 from https://www.msys2.org/, then in MINGW64 shell:

```
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-rust
make
```

Or without make:

```
cargo build --release
```

### linux + nvidia cuda (experimental hybrid backend)

Requires CUDA toolkit (`nvcc`) and NVIDIA driver.

```
cargo build --release --features cuda
./target/release/blocknet-vanity --prefix abc --cuda
```

Using make:

```
make all
./blocknet-vanity --prefix abc --cuda
```

`make all` auto-detects CUDA on Linux when `nvcc` is installed.  
Optional overrides: `make all CUDA=1` (force CUDA) or `make all CUDA=0` (force CPU).

### CUDA optimization policy

CUDA tuning is intentionally prefix-first:

- We optimize for prefix throughput first, since production searches are predominantly prefix-based.
- Small suffix deltas are acceptable when prefix gains are materially larger.
- All CUDA optimization decisions are validated with strict interleaved A/B runs (warmup + cooldown), and results are logged in `PERF_NOTES.md`.

## run

```
./blocknet-vanity --prefix abc
```

### flags

```
--prefix <str>      prefix the address must start with (case-insensitive)
--suffix <str>      suffix the address must end with (case-insensitive)
-t, --threads <N>   number of worker threads (default: CPU core count)
--cuda              request hybrid CUDA backend (requires Linux build with --features cuda)
-o, --output <dir>  directory to save wallet JSON files (default: .)
```

At least one of `--prefix` or `--suffix` is required. Both can be combined. Patterns are limited to 8 characters and must only contain valid base58 characters (no `0`, `O`, `I`, `l`).

### examples

```
./blocknet-vanity --prefix dead
./blocknet-vanity --suffix cat
./blocknet-vanity --prefix ab --suffix cd -t 8 -o wallets/
```

### output

Found wallets are saved as JSON files named after their address:

```json
{
  "address": "deadB7x8...",
  "spend_private_key": "...",
  "spend_public_key": "...",
  "view_private_key": "...",
  "view_public_key": "..."
}
```

## license

BSD 3-Clause. See LICENSE.
