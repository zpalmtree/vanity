#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr char BASE58_ALPHABET[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__device__ __forceinline__ char ascii_lower(char c) {
    if (c >= 'A' && c <= 'Z') {
        return static_cast<char>(c + 32);
    }
    return c;
}

__device__ int encode_base58_64(const uint8_t* input, char* out) {
    uint8_t digits[96];
    int index = 0;

    for (int i = 0; i < 64; ++i) {
        int carry = static_cast<int>(input[i]);
        for (int j = 0; j < index; ++j) {
            carry += static_cast<int>(digits[j]) << 8;
            digits[j] = static_cast<uint8_t>(carry % 58);
            carry /= 58;
        }
        while (carry > 0) {
            digits[index++] = static_cast<uint8_t>(carry % 58);
            carry /= 58;
        }
    }

    int leading_zeros = 0;
    while (leading_zeros < 64 && input[leading_zeros] == 0) {
        leading_zeros++;
    }
    for (int i = 0; i < leading_zeros; ++i) {
        digits[index++] = 0;
    }

    for (int i = 0; i < index; ++i) {
        out[i] = BASE58_ALPHABET[digits[index - 1 - i]];
    }
    return index;
}

__global__ void match_kernel(
    const uint8_t* inputs_64,
    int batch_size,
    const char* prefix_lower,
    int prefix_len,
    const char* suffix_lower,
    int suffix_len,
    uint8_t* out_flags
) {
    int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= batch_size) {
        return;
    }

    const uint8_t* input = inputs_64 + (tid * 64);
    char addr[96];
    int addr_len = encode_base58_64(input, addr);

    bool prefix_ok = true;
    if (prefix_len > 0) {
        if (prefix_len > addr_len) {
            prefix_ok = false;
        } else {
            for (int i = 0; i < prefix_len; ++i) {
                if (ascii_lower(addr[i]) != prefix_lower[i]) {
                    prefix_ok = false;
                    break;
                }
            }
        }
    }

    bool suffix_ok = true;
    if (suffix_len > 0) {
        if (suffix_len > addr_len) {
            suffix_ok = false;
        } else {
            int start = addr_len - suffix_len;
            for (int i = 0; i < suffix_len; ++i) {
                if (ascii_lower(addr[start + i]) != suffix_lower[i]) {
                    suffix_ok = false;
                    break;
                }
            }
        }
    }

    out_flags[tid] = (prefix_ok && suffix_ok) ? 1 : 0;
}

}  // namespace

extern "C" int cuda_match_batch(
    const uint8_t* inputs_64,
    int batch_size,
    const char* prefix_lower,
    int prefix_len,
    const char* suffix_lower,
    int suffix_len,
    uint8_t* out_flags
) {
    if (inputs_64 == nullptr || out_flags == nullptr || batch_size <= 0) {
        return 10;
    }

    size_t inputs_size = static_cast<size_t>(batch_size) * 64;
    size_t flags_size = static_cast<size_t>(batch_size);
    size_t prefix_size = static_cast<size_t>(prefix_len > 0 ? prefix_len : 1);
    size_t suffix_size = static_cast<size_t>(suffix_len > 0 ? suffix_len : 1);

    uint8_t* d_inputs = nullptr;
    uint8_t* d_flags = nullptr;
    char* d_prefix = nullptr;
    char* d_suffix = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_inputs, inputs_size) != cudaSuccess) { rc = 11; goto cleanup; }
    if (cudaMalloc(&d_flags, flags_size) != cudaSuccess) { rc = 12; goto cleanup; }
    if (cudaMalloc(&d_prefix, prefix_size) != cudaSuccess) { rc = 13; goto cleanup; }
    if (cudaMalloc(&d_suffix, suffix_size) != cudaSuccess) { rc = 14; goto cleanup; }

    if (cudaMemcpy(d_inputs, inputs_64, inputs_size, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 21; goto cleanup; }
    if (prefix_len > 0 &&
        cudaMemcpy(d_prefix, prefix_lower, static_cast<size_t>(prefix_len), cudaMemcpyHostToDevice) != cudaSuccess) { rc = 22; goto cleanup; }
    if (suffix_len > 0 &&
        cudaMemcpy(d_suffix, suffix_lower, static_cast<size_t>(suffix_len), cudaMemcpyHostToDevice) != cudaSuccess) { rc = 23; goto cleanup; }

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    match_kernel<<<blocks, threads>>>(
        d_inputs,
        batch_size,
        d_prefix,
        prefix_len,
        d_suffix,
        suffix_len,
        d_flags
    );

    if (cudaDeviceSynchronize() != cudaSuccess) { rc = 31; goto cleanup; }
    if (cudaMemcpy(out_flags, d_flags, flags_size, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 32; goto cleanup; }

cleanup:
    if (d_inputs != nullptr) cudaFree(d_inputs);
    if (d_flags != nullptr) cudaFree(d_flags);
    if (d_prefix != nullptr) cudaFree(d_prefix);
    if (d_suffix != nullptr) cudaFree(d_suffix);
    return rc;
}
