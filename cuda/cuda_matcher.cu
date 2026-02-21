#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ============================================================================
// Field arithmetic for GF(2^255-19) using radix-2^51 representation
// 5 limbs, each up to 51 bits (with lazy reduction allowing temporary overflow)
// ============================================================================

typedef unsigned long long u64;
typedef unsigned __int128 u128;

// Address format:
// base58(spend_pub[32] || view_pub[32] || checksum4)
// checksum4 = first 4 bytes of SHA3-256("blocknet_stealth_address_checksum" || "blocknet_mainnet" || spend_pub || view_pub)
#define ADDRESS_PUBKEY_BYTES 64
#define ADDRESS_CHECKSUM_BYTES 4
#define ADDRESS_TOTAL_BYTES (ADDRESS_PUBKEY_BYTES + ADDRESS_CHECKSUM_BYTES)

struct fe25519 {
    u64 v[5];
};

// p = 2^255 - 19
// In radix-2^51: [2251799813685229, 2251799813685247, 2251799813685247, 2251799813685247, 2251799813685247]
// That is: (2^51 - 19), (2^51 - 1), (2^51 - 1), (2^51 - 1), (2^51 - 1)

__device__ __forceinline__ void fe_zero(fe25519* h) {
    h->v[0] = 0; h->v[1] = 0; h->v[2] = 0; h->v[3] = 0; h->v[4] = 0;
}

__device__ __forceinline__ void fe_one(fe25519* h) {
    h->v[0] = 1; h->v[1] = 0; h->v[2] = 0; h->v[3] = 0; h->v[4] = 0;
}

__device__ __forceinline__ void fe_copy(fe25519* h, const fe25519* f) {
    h->v[0] = f->v[0]; h->v[1] = f->v[1]; h->v[2] = f->v[2];
    h->v[3] = f->v[3]; h->v[4] = f->v[4];
}

__device__ __forceinline__ void fe_add(fe25519* h, const fe25519* f, const fe25519* g) {
    h->v[0] = f->v[0] + g->v[0];
    h->v[1] = f->v[1] + g->v[1];
    h->v[2] = f->v[2] + g->v[2];
    h->v[3] = f->v[3] + g->v[3];
    h->v[4] = f->v[4] + g->v[4];
}

// Subtraction with bias to keep positive: h = f - g + 2p
__device__ __forceinline__ void fe_sub(fe25519* h, const fe25519* f, const fe25519* g) {
    // Add 2p to avoid underflow: 2p in radix-2^51 limbs
    // 2p = [2*(2^51-19), 2*(2^51-1), 2*(2^51-1), 2*(2^51-1), 2*(2^51-1)]
    const u64 mask = (1ULL << 51) - 1;
    h->v[0] = (f->v[0] + 2ULL * ((1ULL << 51) - 19)) - g->v[0];
    h->v[1] = (f->v[1] + 2ULL * mask) - g->v[1];
    h->v[2] = (f->v[2] + 2ULL * mask) - g->v[2];
    h->v[3] = (f->v[3] + 2ULL * mask) - g->v[3];
    h->v[4] = (f->v[4] + 2ULL * mask) - g->v[4];
}

// Carry propagation to keep limbs roughly 51 bits
__device__ __forceinline__ void fe_carry(fe25519* h) {
    const u64 mask = (1ULL << 51) - 1;
    u64 c;
    c = h->v[0] >> 51; h->v[0] &= mask; h->v[1] += c;
    c = h->v[1] >> 51; h->v[1] &= mask; h->v[2] += c;
    c = h->v[2] >> 51; h->v[2] &= mask; h->v[3] += c;
    c = h->v[3] >> 51; h->v[3] &= mask; h->v[4] += c;
    c = h->v[4] >> 51; h->v[4] &= mask; h->v[0] += c * 19;
    // One more round for the wrap-around carry
    c = h->v[0] >> 51; h->v[0] &= mask; h->v[1] += c;
}

// Multiplication: h = f * g mod p
__device__ void fe_mul(fe25519* h, const fe25519* f, const fe25519* g) {
    u64 f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    u64 g0 = g->v[0], g1 = g->v[1], g2 = g->v[2], g3 = g->v[3], g4 = g->v[4];

    u64 g1_19 = g1 * 19, g2_19 = g2 * 19, g3_19 = g3 * 19, g4_19 = g4 * 19;

    u128 h0 = (u128)f0*g0 + (u128)f1*g4_19 + (u128)f2*g3_19 + (u128)f3*g2_19 + (u128)f4*g1_19;
    u128 h1 = (u128)f0*g1 + (u128)f1*g0   + (u128)f2*g4_19 + (u128)f3*g3_19 + (u128)f4*g2_19;
    u128 h2 = (u128)f0*g2 + (u128)f1*g1   + (u128)f2*g0    + (u128)f3*g4_19 + (u128)f4*g3_19;
    u128 h3 = (u128)f0*g3 + (u128)f1*g2   + (u128)f2*g1    + (u128)f3*g0    + (u128)f4*g4_19;
    u128 h4 = (u128)f0*g4 + (u128)f1*g3   + (u128)f2*g2    + (u128)f3*g1    + (u128)f4*g0;

    const u64 mask = (1ULL << 51) - 1;
    u64 c;

    c = (u64)(h0 >> 51); h->v[0] = (u64)h0 & mask; h1 += c;
    c = (u64)(h1 >> 51); h->v[1] = (u64)h1 & mask; h2 += c;
    c = (u64)(h2 >> 51); h->v[2] = (u64)h2 & mask; h3 += c;
    c = (u64)(h3 >> 51); h->v[3] = (u64)h3 & mask; h4 += c;
    c = (u64)(h4 >> 51); h->v[4] = (u64)h4 & mask;

    h->v[0] += c * 19;
    c = h->v[0] >> 51; h->v[0] &= mask; h->v[1] += c;
}

// Squaring: h = f^2 mod p (optimized: uses symmetry to halve multiplications)
__device__ void fe_sq(fe25519* h, const fe25519* f) {
    u64 f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    u64 f0_2 = f0 * 2, f1_2 = f1 * 2;

    // h0 = f0^2 + 2*19*f1*f4 + 2*19*f2*f3
    u128 h0 = (u128)f0*f0     + (u128)(f1*38)*f4   + (u128)(f2*38)*f3;
    // h1 = 2*f0*f1 + 2*19*f2*f4 + 19*f3^2
    u128 h1 = (u128)f0_2*f1   + (u128)(f2*38)*f4   + (u128)(f3*19)*f3;
    // h2 = 2*f0*f2 + f1^2 + 2*19*f3*f4
    u128 h2 = (u128)f0_2*f2   + (u128)f1*f1         + (u128)(f3*38)*f4;
    // h3 = 2*f0*f3 + 2*f1*f2 + 19*f4^2
    u128 h3 = (u128)f0_2*f3   + (u128)f1_2*f2       + (u128)(f4*19)*f4;
    // h4 = 2*f0*f4 + 2*f1*f3 + f2^2
    u128 h4 = (u128)f0_2*f4   + (u128)f1_2*f3       + (u128)f2*f2;

    const u64 mask = (1ULL << 51) - 1;
    u64 c;

    c = (u64)(h0 >> 51); h->v[0] = (u64)h0 & mask; h1 += c;
    c = (u64)(h1 >> 51); h->v[1] = (u64)h1 & mask; h2 += c;
    c = (u64)(h2 >> 51); h->v[2] = (u64)h2 & mask; h3 += c;
    c = (u64)(h3 >> 51); h->v[3] = (u64)h3 & mask; h4 += c;
    c = (u64)(h4 >> 51); h->v[4] = (u64)h4 & mask;

    h->v[0] += c * 19;
    c = h->v[0] >> 51; h->v[0] &= mask; h->v[1] += c;
}

// Multiply by small constant: h = f * small
__device__ __forceinline__ void fe_mul_small(fe25519* h, const fe25519* f, u64 small) {
    const u64 mask = (1ULL << 51) - 1;
    u128 t;
    u64 c;

    t = (u128)f->v[0] * small;              h->v[0] = (u64)t & mask; c = (u64)(t >> 51);
    t = (u128)f->v[1] * small + c;          h->v[1] = (u64)t & mask; c = (u64)(t >> 51);
    t = (u128)f->v[2] * small + c;          h->v[2] = (u64)t & mask; c = (u64)(t >> 51);
    t = (u128)f->v[3] * small + c;          h->v[3] = (u64)t & mask; c = (u64)(t >> 51);
    t = (u128)f->v[4] * small + c;          h->v[4] = (u64)t & mask; c = (u64)(t >> 51);

    h->v[0] += c * 19;
    c = h->v[0] >> 51; h->v[0] &= mask; h->v[1] += c;
}

// Negation: h = -f mod p = 2p - f
__device__ __forceinline__ void fe_neg(fe25519* h, const fe25519* f) {
    fe25519 zero;
    fe_zero(&zero);
    fe_sub(h, &zero, f);
}

// Full reduction to [0, p)
__device__ void fe_reduce(fe25519* h) {
    fe_carry(h);

    const u64 mask = (1ULL << 51) - 1;
    // Now check if h >= p: p = [2^51-19, 2^51-1, 2^51-1, 2^51-1, 2^51-1]
    // Subtract p and check if result is non-negative
    u64 q = (h->v[0] + 19) >> 51;
    q = (h->v[1] + q) >> 51;
    q = (h->v[2] + q) >> 51;
    q = (h->v[3] + q) >> 51;
    q = (h->v[4] + q) >> 51;

    h->v[0] += q * 19;
    u64 c;
    c = h->v[0] >> 51; h->v[0] &= mask; h->v[1] += c;
    c = h->v[1] >> 51; h->v[1] &= mask; h->v[2] += c;
    c = h->v[2] >> 51; h->v[2] &= mask; h->v[3] += c;
    c = h->v[3] >> 51; h->v[3] &= mask; h->v[4] += c;
    h->v[4] &= mask;
}

// Serialize field element to 32 bytes (little-endian)
__device__ void fe_tobytes(uint8_t* s, const fe25519* f) {
    fe25519 t;
    fe_copy(&t, f);
    fe_reduce(&t);

    // Pack 5 x 51-bit limbs into 4 x u64 (256 bits LE), then store as bytes
    // Limb layout: v0[0:50], v1[51:101], v2[102:152], v3[153:203], v4[204:254]
    u64 v0 = t.v[0], v1 = t.v[1], v2 = t.v[2], v3 = t.v[3], v4 = t.v[4];

    u64 lo0 = v0       | (v1 << 51);          // bits 0-63
    u64 lo1 = (v1 >> 13) | (v2 << 38);        // bits 64-127
    u64 lo2 = (v2 >> 26) | (v3 << 25);        // bits 128-191
    u64 lo3 = (v3 >> 39) | (v4 << 12);        // bits 192-255

    // Store as little-endian bytes
    for (int i = 0; i < 8; i++) s[i]      = (uint8_t)(lo0 >> (i * 8));
    for (int i = 0; i < 8; i++) s[8 + i]  = (uint8_t)(lo1 >> (i * 8));
    for (int i = 0; i < 8; i++) s[16 + i] = (uint8_t)(lo2 >> (i * 8));
    for (int i = 0; i < 8; i++) s[24 + i] = (uint8_t)(lo3 >> (i * 8));
}

// Deserialize 32 bytes (little-endian) to field element
__device__ void fe_frombytes(fe25519* h, const uint8_t* s) {
    // Read as a 256-bit little-endian integer, split into 51-bit limbs
    u64 lo0 = 0, lo1 = 0, lo2 = 0, lo3 = 0;
    for (int i = 7; i >= 0; i--) lo0 = (lo0 << 8) | s[i];
    for (int i = 15; i >= 8; i--) lo1 = (lo1 << 8) | s[i];
    for (int i = 23; i >= 16; i--) lo2 = (lo2 << 8) | s[i];
    for (int i = 31; i >= 24; i--) lo3 = (lo3 << 8) | s[i];

    const u64 mask = (1ULL << 51) - 1;
    // 256 bits across lo0(64) lo1(64) lo2(64) lo3(64)
    // limb 0: bits 0..50
    h->v[0] = lo0 & mask;
    // limb 1: bits 51..101
    h->v[1] = ((lo0 >> 51) | (lo1 << 13)) & mask;
    // limb 2: bits 102..152
    h->v[2] = ((lo1 >> 38) | (lo2 << 26)) & mask;
    // limb 3: bits 153..203
    h->v[3] = ((lo2 >> 25) | (lo3 << 39)) & mask;
    // limb 4: bits 204..254
    h->v[4] = (lo3 >> 12) & mask;
}

// Check if field element is negative (odd) - reduced limb check avoids fe_tobytes
__device__ __forceinline__ int fe_isneg(const fe25519* f) {
    fe25519 t;
    fe_copy(&t, f);
    fe_reduce(&t);
    return (int)(t.v[0] & 1);
}

// Check if field element is zero - reduced limb check avoids fe_tobytes
__device__ int fe_iszero(const fe25519* f) {
    fe25519 t;
    fe_copy(&t, f);
    fe_reduce(&t);
    u64 r = t.v[0] | t.v[1] | t.v[2] | t.v[3] | t.v[4];
    return r == 0;
}

// Conditional swap: if b != 0, swap f and g
__device__ __forceinline__ void fe_cswap(fe25519* f, fe25519* g, int b) {
    u64 mask = (u64)(-(int64_t)b);
    for (int i = 0; i < 5; i++) {
        u64 x = (f->v[i] ^ g->v[i]) & mask;
        f->v[i] ^= x;
        g->v[i] ^= x;
    }
}

// Conditional move: if b != 0, f = g
__device__ __forceinline__ void fe_cmov(fe25519* f, const fe25519* g, int b) {
    u64 mask = (u64)(-(int64_t)b);
    for (int i = 0; i < 5; i++) {
        f->v[i] ^= (f->v[i] ^ g->v[i]) & mask;
    }
}

// Absolute value: if f is negative (odd), negate it
__device__ __forceinline__ void fe_abs(fe25519* h, const fe25519* f) {
    fe25519 neg;
    fe_neg(&neg, f);
    fe_copy(h, f);
    fe_cmov(h, &neg, fe_isneg(f));
}

// f^(2^n) via repeated squaring
__device__ void fe_sq_n(fe25519* h, const fe25519* f, int n) {
    fe_sq(h, f);
    for (int i = 1; i < n; i++) {
        fe_sq(h, h);
    }
}

// Compute f^(p-2) = f^(2^255-21) for inversion
__device__ void fe_invert(fe25519* h, const fe25519* f) {
    fe25519 t0, t1, t2, t3;

    fe_sq(&t0, f);           // t0 = f^2
    fe_sq_n(&t1, &t0, 2);    // t1 = f^8
    fe_mul(&t1, f, &t1);     // t1 = f^9
    fe_mul(&t0, &t0, &t1);   // t0 = f^11
    fe_sq(&t2, &t0);         // t2 = f^22
    fe_mul(&t1, &t1, &t2);   // t1 = f^(2^5-1) = f^31
    fe_sq_n(&t2, &t1, 5);    // t2 = f^(2^10-32)
    fe_mul(&t1, &t2, &t1);   // t1 = f^(2^10-1)
    fe_sq_n(&t2, &t1, 10);   // t2 = f^(2^20-1024)
    fe_mul(&t2, &t2, &t1);   // t2 = f^(2^20-1)
    fe_sq_n(&t3, &t2, 20);   // t3 = f^(2^40-2^20)
    fe_mul(&t2, &t3, &t2);   // t2 = f^(2^40-1)
    fe_sq_n(&t2, &t2, 10);   // t2 = f^(2^50-1024)
    fe_mul(&t1, &t2, &t1);   // t1 = f^(2^50-1)
    fe_sq_n(&t2, &t1, 50);   // t2 = f^(2^100-2^50)
    fe_mul(&t2, &t2, &t1);   // t2 = f^(2^100-1)
    fe_sq_n(&t3, &t2, 100);  // t3 = f^(2^200-2^100)
    fe_mul(&t2, &t3, &t2);   // t2 = f^(2^200-1)
    fe_sq_n(&t2, &t2, 50);   // t2 = f^(2^250-2^50)
    fe_mul(&t1, &t2, &t1);   // t1 = f^(2^250-1)
    fe_sq_n(&t1, &t1, 5);    // t1 = f^(2^255-32)
    fe_mul(h, &t1, &t0);     // h  = f^(2^255-21)
}

// Compute f^((p-5)/8) = f^(2^252-3)
// Used for square root computation
__device__ void fe_pow22523(fe25519* h, const fe25519* f) {
    fe25519 t0, t1, t2;

    fe_sq(&t0, f);            // t0 = f^2
    fe_sq_n(&t1, &t0, 2);     // t1 = f^8
    fe_mul(&t1, f, &t1);      // t1 = f^9
    fe_mul(&t0, &t0, &t1);    // t0 = f^11
    fe_sq(&t0, &t0);          // t0 = f^22
    fe_mul(&t0, &t1, &t0);    // t0 = f^31 = f^(2^5-1)
    fe_sq_n(&t1, &t0, 5);     // t1 = f^(2^10-32)
    fe_mul(&t0, &t1, &t0);    // t0 = f^(2^10-1)
    fe_sq_n(&t1, &t0, 10);    // t1 = f^(2^20-1024)
    fe_mul(&t1, &t1, &t0);    // t1 = f^(2^20-1)
    fe_sq_n(&t2, &t1, 20);    // t2 = f^(2^40-2^20)
    fe_mul(&t1, &t2, &t1);    // t1 = f^(2^40-1)
    fe_sq_n(&t1, &t1, 10);    // t1 = f^(2^50-1024)
    fe_mul(&t0, &t1, &t0);    // t0 = f^(2^50-1)
    fe_sq_n(&t1, &t0, 50);    // t1 = f^(2^100-2^50)
    fe_mul(&t1, &t1, &t0);    // t1 = f^(2^100-1)
    fe_sq_n(&t2, &t1, 100);   // t2 = f^(2^200-2^100)
    fe_mul(&t1, &t2, &t1);    // t1 = f^(2^200-1)
    fe_sq_n(&t1, &t1, 50);    // t1 = f^(2^250-2^50)
    fe_mul(&t0, &t1, &t0);    // t0 = f^(2^250-1)
    fe_sq_n(&t0, &t0, 2);     // t0 = f^(2^252-4)
    fe_mul(h, &t0, f);        // h  = f^(2^252-3)
}

// fe_equal: returns 1 if f == g - uses subtraction + single reduce
__device__ int fe_equal(const fe25519* f, const fe25519* g) {
    fe25519 diff;
    fe_sub(&diff, f, g);   // diff = f - g + 2p (always positive)
    fe_reduce(&diff);
    u64 d = diff.v[0] | diff.v[1] | diff.v[2] | diff.v[3] | diff.v[4];
    return d == 0;
}


// ============================================================================
// Constants for Curve25519 / Ristretto255
// ============================================================================

// sqrt(-1) mod p
__device__ __constant__ fe25519 SQRT_M1 = {{
    1718705420411056ULL, 234908883556509ULL, 2233514472574048ULL,
    2117202627021982ULL, 765476049583133ULL
}};

// d = -121665/121666 mod p (Edwards curve parameter)
__device__ __constant__ fe25519 EDWARDS_D = {{
    929955233495203ULL, 466365720129213ULL, 1662059464998953ULL,
    2033849074728123ULL, 1442794654840575ULL
}};

// 2*d
__device__ __constant__ fe25519 EDWARDS_D2 = {{
    1859910466990425ULL, 932731440258426ULL, 1072319116312658ULL,
    1815898335770999ULL, 633789495995903ULL
}};

// 1/sqrt(a-d) where a=-1, so 1/sqrt(-1-d)
// From dalek: INVSQRT_A_MINUS_D
__device__ __constant__ fe25519 INVSQRT_A_MINUS_D = {{
    278908739862762ULL, 821645201101625ULL, 8113234426968ULL,
    1777959178193151ULL, 2118520810568447ULL
}};

// ============================================================================
// Edwards Point (extended coordinates: X:Y:Z:T where x=X/Z, y=Y/Z, xy=T/Z)
// ============================================================================

struct ge25519_p3 {
    fe25519 X, Y, Z, T;
};

// R = P + Q (unified addition, extended coordinates)
// Using the add-2008-hwcd formula for a=-1 twisted Edwards curves
__device__ void ge_add(ge25519_p3* R, const ge25519_p3* P, const ge25519_p3* Q) {
    fe25519 A, B, C, D, E, F, G, H;

    // A = (Y1-X1)*(Y2-X2)
    fe25519 t0, t1;
    fe_sub(&t0, &P->Y, &P->X);
    fe_sub(&t1, &Q->Y, &Q->X);
    fe_mul(&A, &t0, &t1);

    // B = (Y1+X1)*(Y2+X2)
    fe_add(&t0, &P->Y, &P->X);
    fe_add(&t1, &Q->Y, &Q->X);
    fe_mul(&B, &t0, &t1);

    // C = T1*2d*T2
    fe_mul(&C, &P->T, &Q->T);
    fe_mul(&C, &C, &EDWARDS_D2);

    // D = Z1*2*Z2
    fe_mul(&D, &P->Z, &Q->Z);
    fe_add(&D, &D, &D);

    // E = B - A
    fe_sub(&E, &B, &A);
    // F = D - C
    fe_sub(&F, &D, &C);
    // G = D + C
    fe_add(&G, &D, &C);
    // H = B + A
    fe_add(&H, &B, &A);

    fe_mul(&R->X, &E, &F);
    fe_mul(&R->Y, &G, &H);
    fe_mul(&R->Z, &F, &G);
    fe_mul(&R->T, &E, &H);
}

// Identity point: (0:1:1:0)
__device__ __forceinline__ void ge_identity(ge25519_p3* P) {
    fe_zero(&P->X);
    fe_one(&P->Y);
    fe_one(&P->Z);
    fe_zero(&P->T);
}

// ============================================================================
// Ristretto255 Compression
// ============================================================================

// SQRT_RATIO_M1: compute sqrt(u/v) or sqrt(i*u/v)
// Returns (was_nonzero_square, result)
__device__ int sqrt_ratio_m1(fe25519* result, const fe25519* u, const fe25519* v) {
    fe25519 v3, v7, r, check, neg_u, neg_u_i;

    // v^3 = v^2 * v
    fe_sq(&v3, v);
    fe_mul(&v3, &v3, v);

    // v^7 = v^3 * v^3 * v = (v^3)^2 * v
    fe_sq(&v7, &v3);
    fe_mul(&v7, &v7, v);

    // r = u * v^3 * (u * v^7)^((p-5)/8)
    fe25519 uv7;
    fe_mul(&uv7, u, &v7);
    fe_pow22523(&r, &uv7);
    fe_mul(&r, &r, &v3);
    fe_mul(&r, &r, u);

    // check = v * r^2
    fe_sq(&check, &r);
    fe_mul(&check, &check, v);

    // neg_u = -u
    fe_neg(&neg_u, u);

    // neg_u_i = -u * sqrt(-1)
    fe_mul(&neg_u_i, &neg_u, &SQRT_M1);

    int correct_sign = fe_equal(&check, u);
    int flipped_sign = fe_equal(&check, &neg_u);
    int flipped_sign_i = fe_equal(&check, &neg_u_i);

    // If flipped, multiply r by sqrt(-1)
    fe25519 r_prime;
    fe_mul(&r_prime, &r, &SQRT_M1);
    fe_cmov(&r, &r_prime, flipped_sign | flipped_sign_i);

    // Choose non-negative (even) square root
    fe_abs(&r, &r);

    fe_copy(result, &r);
    return correct_sign | flipped_sign;
}

// Ristretto compression: point -> 32 bytes
__device__ void ristretto_encode(uint8_t* out, const ge25519_p3* P) {
    fe25519 u1, u2, inv, den1, den2, z_inv;
    fe25519 ix0, iy0, ench_den;
    fe25519 x, y, s;

    // u1 = (Z + Y) * (Z - Y)
    fe25519 t0, t1;
    fe_add(&t0, &P->Z, &P->Y);
    fe_sub(&t1, &P->Z, &P->Y);
    fe_mul(&u1, &t0, &t1);

    // u2 = X * Y
    fe_mul(&u2, &P->X, &P->Y);

    // inv = invsqrt(u1 * u2^2) via SQRT_RATIO_M1(1, u1*u2^2)
    fe25519 u2_sq, u1_u2sq, one;
    fe_sq(&u2_sq, &u2);
    fe_mul(&u1_u2sq, &u1, &u2_sq);
    fe_one(&one);
    sqrt_ratio_m1(&inv, &one, &u1_u2sq);

    // den1 = inv * u1
    fe_mul(&den1, &inv, &u1);

    // den2 = inv * u2
    fe_mul(&den2, &inv, &u2);

    // z_inv = den1 * den2 * T
    fe_mul(&z_inv, &den1, &den2);
    fe_mul(&z_inv, &z_inv, &P->T);

    // ix0 = X * sqrt(-1)
    fe_mul(&ix0, &P->X, &SQRT_M1);

    // iy0 = Y * sqrt(-1)
    fe_mul(&iy0, &P->Y, &SQRT_M1);

    // enchanted_denominator = den1 * INVSQRT_A_MINUS_D
    fe_mul(&ench_den, &den1, &INVSQRT_A_MINUS_D);

    // rotate = IS_NEGATIVE(T * z_inv)
    fe25519 t_zinv;
    fe_mul(&t_zinv, &P->T, &z_inv);
    int rotate = fe_isneg(&t_zinv);

    // Conditional rotation
    fe_copy(&x, &P->X);
    fe_cmov(&x, &iy0, rotate);

    fe_copy(&y, &P->Y);
    fe_cmov(&y, &ix0, rotate);

    fe25519 den_inv;
    fe_copy(&den_inv, &den2);
    fe_cmov(&den_inv, &ench_den, rotate);

    // if IS_NEGATIVE(x * z_inv): y = -y
    fe25519 x_zinv;
    fe_mul(&x_zinv, &x, &z_inv);
    int neg_y = fe_isneg(&x_zinv);
    fe25519 neg_y_val;
    fe_neg(&neg_y_val, &y);
    fe_cmov(&y, &neg_y_val, neg_y);

    // s = |den_inv * (Z - y)|
    fe_sub(&t0, &P->Z, &y);
    fe_mul(&s, &den_inv, &t0);
    fe_abs(&s, &s);

    // Encode s as 32 bytes
    fe_tobytes(out, &s);
}


// ============================================================================
// Base58 encoding (same as before but now after GPU-side key generation)
// ============================================================================

__device__ __constant__ char BASE58_ALPHABET_DEVICE[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
// SHA3-256 one-block absorb base lanes for:
// "blocknet_stealth_address_checksumblocknet_mainnet" || spend_pub || view_pub || 0x06 || ... || 0x80
// with spend_pub/view_pub bytes left zero and padding bits pre-set.
__device__ __constant__ u64 ADDRESS_CHECKSUM_BASE_LANES[17] = {
    0x74656e6b636f6c62ULL, 0x68746c616574735fULL, 0x737365726464615fULL,
    0x75736b636568635fULL, 0x656e6b636f6c626dULL, 0x656e6e69616d5f74ULL,
    0x0000000000000074ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000600ULL,
    0x0000000000000000ULL, 0x8000000000000000ULL
};

__device__ __forceinline__ char ascii_lower(char c) {
    if (c >= 'A' && c <= 'Z') return (char)(c + 32);
    return c;
}

__device__ __forceinline__ u64 rotl64(u64 x, int s) {
    return (x << s) | (x >> (64 - s));
}

__device__ __constant__ u64 KECCAKF_RNDC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __constant__ int KECCAKF_ROTC[24] = {
    1,  3,  6, 10, 15, 21, 28, 36,
    45, 55,  2, 14, 27, 41, 56,  8,
    25, 43, 62, 18, 39, 61, 20, 44
};

__device__ __constant__ int KECCAKF_PILN[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,
     8, 21, 24,  4, 15, 23, 19, 13,
    12,  2, 20, 14, 22,  9,  6,  1
};

__device__ __forceinline__ u64 load64_le(const uint8_t* src) {
    u64 v = 0;
    for (int i = 0; i < 8; i++) {
        v |= ((u64)src[i]) << (8 * i);
    }
    return v;
}

__device__ __forceinline__ u64 load64_be(const uint8_t* src) {
    u64 v = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v = (v << 8) | (u64)src[i];
    }
    return v;
}

__device__ void keccakf(u64 st[25]) {
    u64 bc[5];
    for (int round = 0; round < 24; round++) {
        // Theta
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for (int i = 0; i < 5; i++) {
            u64 t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            st[i] ^= t;
            st[i + 5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        // Rho + Pi
        u64 t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAKF_PILN[i];
            u64 tmp = st[j];
            st[j] = rotl64(t, KECCAKF_ROTC[i]);
            t = tmp;
        }

        // Chi
        for (int j = 0; j < 25; j += 5) {
            u64 s0 = st[j + 0];
            u64 s1 = st[j + 1];
            u64 s2 = st[j + 2];
            u64 s3 = st[j + 3];
            u64 s4 = st[j + 4];
            st[j + 0] = s0 ^ ((~s1) & s2);
            st[j + 1] = s1 ^ ((~s2) & s3);
            st[j + 2] = s2 ^ ((~s3) & s4);
            st[j + 3] = s3 ^ ((~s4) & s0);
            st[j + 4] = s4 ^ ((~s0) & s1);
        }

        // Iota
        st[0] ^= KECCAKF_RNDC[round];
    }
}

// checksum4 = first 4 bytes of SHA3-256(tag || network_id || spend_pub || view_pub)
__device__ __forceinline__ void address_checksum4(
    const uint8_t* spend_pub,
    const uint8_t* view_pub,
    uint8_t out4[4]
) {
    u64 st[25];
    #pragma unroll
    for (int i = 0; i < 17; i++) st[i] = ADDRESS_CHECKSUM_BASE_LANES[i];
    #pragma unroll
    for (int i = 17; i < 25; i++) st[i] = 0;

    // spend_pub starts at byte offset 49 (lane 6, shift 8)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 chunk = load64_le(spend_pub + i * 8);
        st[6 + i] ^= (chunk << 8);
        st[7 + i] ^= (chunk >> 56);
    }
    // view_pub starts at byte offset 81 (lane 10, shift 8)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 chunk = load64_le(view_pub + i * 8);
        st[10 + i] ^= (chunk << 8);
        st[11 + i] ^= (chunk >> 56);
    }

    keccakf(st);

    out4[0] = (uint8_t)(st[0] & 0xFF);
    out4[1] = (uint8_t)((st[0] >> 8) & 0xFF);
    out4[2] = (uint8_t)((st[0] >> 16) & 0xFF);
    out4[3] = (uint8_t)((st[0] >> 24) & 0xFF);
}

// Base58 using radix-58^4 = 11,316,496 for ~4x fewer inner-loop iterations
// Each "group" holds 4 base-58 digits packed into a u32.
// Max intermediate value during carry: 255 + 11316495 * 256 = 2,896,982,975 < 2^32
#define BASE58_POW4 11316496U

__device__ int encode_base58(const uint8_t* input, int input_len, char* out) {
    uint32_t groups[28];  // enough for 68-byte input with margin
    int ngroups = 0;

    for (int i = 0; i < input_len; ++i) {
        uint32_t carry = (uint32_t)input[i];
        for (int j = 0; j < ngroups; ++j) {
            uint64_t acc = (uint64_t)groups[j] * 256 + carry;
            groups[j] = (uint32_t)(acc % BASE58_POW4);
            carry = (uint32_t)(acc / BASE58_POW4);
        }
        while (carry > 0) {
            groups[ngroups++] = (uint32_t)(carry % BASE58_POW4);
            carry /= BASE58_POW4;
        }
    }

    // Convert groups to individual base-58 digits (little-endian within each group)
    uint8_t digits[128];
    int ndigits = 0;
    for (int g = 0; g < ngroups - 1; ++g) {
        uint32_t val = groups[g];
        digits[ndigits++] = (uint8_t)(val % 58); val /= 58;
        digits[ndigits++] = (uint8_t)(val % 58); val /= 58;
        digits[ndigits++] = (uint8_t)(val % 58); val /= 58;
        digits[ndigits++] = (uint8_t)(val);
    }
    // Last group: don't emit leading zeros
    if (ngroups > 0) {
        uint32_t val = groups[ngroups - 1];
        while (val > 0) {
            digits[ndigits++] = (uint8_t)(val % 58);
            val /= 58;
        }
    }

    // Count leading zero bytes -> leading '1's
    int leading_zeros = 0;
    while (leading_zeros < input_len && input[leading_zeros] == 0) leading_zeros++;
    for (int i = 0; i < leading_zeros; ++i) digits[ndigits++] = 0;

    // Reverse into output with alphabet mapping
    for (int i = 0; i < ndigits; ++i)
        out[i] = BASE58_ALPHABET_DEVICE[digits[ndigits - 1 - i]];
    return ndigits;
}


// ============================================================================
// Legacy kernel: match pre-generated keys (kept for compatibility)
// ============================================================================

__global__ void match_kernel(
    const uint8_t* inputs_64, int batch_size,
    const char* prefix_lower, int prefix_len,
    const char* suffix_lower, int suffix_len,
    uint8_t* out_flags
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= batch_size) return;

    const uint8_t* input = inputs_64 + (tid * ADDRESS_PUBKEY_BYTES);
    uint8_t combined[ADDRESS_TOTAL_BYTES];
    for (int i = 0; i < ADDRESS_PUBKEY_BYTES; i++) combined[i] = input[i];

    uint8_t checksum4[4];
    address_checksum4(input, input + 32, checksum4);
    combined[64] = checksum4[0];
    combined[65] = checksum4[1];
    combined[66] = checksum4[2];
    combined[67] = checksum4[3];

    char addr[128];
    int addr_len = encode_base58(combined, ADDRESS_TOTAL_BYTES, addr);

    bool prefix_ok = true;
    if (prefix_len > 0) {
        if (prefix_len > addr_len) { prefix_ok = false; }
        else {
            for (int i = 0; i < prefix_len; ++i) {
                if (ascii_lower(addr[i]) != prefix_lower[i]) { prefix_ok = false; break; }
            }
        }
    }

    bool suffix_ok = true;
    if (prefix_ok && suffix_len > 0) {
        if (suffix_len > addr_len) { suffix_ok = false; }
        else {
            int start = addr_len - suffix_len;
            for (int i = 0; i < suffix_len; ++i) {
                if (ascii_lower(addr[start + i]) != suffix_lower[i]) { suffix_ok = false; break; }
            }
        }
    }

    out_flags[tid] = (prefix_ok && suffix_ok) ? 1 : 0;
}


// ============================================================================
// NEW: Full GPU vanity kernel - key generation + encoding + matching
// ============================================================================

// Generator multiples table: G, 2G, 4G, ..., 2^23*G  (24 entries)
// Set at init time from host
#define TABLE_SIZE 24

// Each thread computes KEYS_PER_THREAD consecutive keys
#define KEYS_PER_THREAD 8
#define KPT_SHIFT 3  // log2(KEYS_PER_THREAD)

__device__ ge25519_p3 d_gen_table[TABLE_SIZE];
__device__ fe25519 d_view_pub_fe[1];  // not used directly, but view_pub bytes are

__device__ __forceinline__ int cmp_spend_view_split_to_bound(
    const uint8_t* spend_pub,
    const uint8_t* view_pub,
    const uint8_t* bound
) {
    for (int i = 0; i < 32; i++) {
        if (spend_pub[i] < bound[i]) return -1;
        if (spend_pub[i] > bound[i]) return 1;
    }
    for (int i = 0; i < 32; i++) {
        uint8_t b = bound[32 + i];
        if (view_pub[i] < b) return -1;
        if (view_pub[i] > b) return 1;
    }
    return 0;
}

#ifndef VANITY_PREFIX_LAUNCH_MIN_BLOCKS
#define VANITY_PREFIX_LAUNCH_MIN_BLOCKS 1
#endif

#ifndef VANITY_GENERIC_LAUNCH_MIN_BLOCKS
#define VANITY_GENERIC_LAUNCH_MIN_BLOCKS 2
#endif

__device__ __forceinline__ int cmp_checksum4_to_bound(
    const uint8_t* checksum4,
    const uint8_t* bound
) {
    for (int i = 0; i < 4; i++) {
        uint8_t b = bound[64 + i];
        if (checksum4[i] < b) return -1;
        if (checksum4[i] > b) return 1;
    }
    return 0;
}

__device__ __forceinline__ u64 suffix_mod_for_address(
    const uint8_t* spend_pub,
    const uint8_t* view_pub,
    const uint8_t* checksum4,
    u64 suffix_modulus,
    u64 suffix_chunk_mul,
    u64 suffix_tail_mul
) {
    u64 mod_val = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 chunk = load64_be(spend_pub + i * 8);
        mod_val = (u64)(((u128)mod_val * suffix_chunk_mul + chunk) % suffix_modulus);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 chunk = load64_be(view_pub + i * 8);
        mod_val = (u64)(((u128)mod_val * suffix_chunk_mul + chunk) % suffix_modulus);
    }

    u64 checksum_chunk = ((u64)checksum4[0] << 24)
                       | ((u64)checksum4[1] << 16)
                       | ((u64)checksum4[2] << 8)
                       | (u64)checksum4[3];

    return (u64)(((u128)mod_val * suffix_tail_mul + checksum_chunk) % suffix_modulus);
}

__global__ void __launch_bounds__(256, VANITY_GENERIC_LAUNCH_MIN_BLOCKS) vanity_kernel(
    // Starting point P (in extended Edwards coordinates, as 5 u64 limbs each)
    const u64* start_X, const u64* start_Y, const u64* start_Z, const u64* start_T,
    // View public key (compressed, 32 bytes) - constant across batch
    const uint8_t* view_pub,
    // Range-based matching (replaces base58 encoding)
    const uint8_t* prefix_ranges, int num_prefix_ranges,
    u64 suffix_modulus, const u64* suffix_targets, int num_suffix_targets,
    u64 suffix_chunk_mul, u64 suffix_tail_mul,
    // Batch: total number of keys (must be multiple of KEYS_PER_THREAD)
    int batch_size,
    // Output: flags (1=match, 0=no match), indexed by key_index [0..batch_size)
    uint8_t* out_flags
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int num_threads = batch_size >> KPT_SHIFT;  // batch_size / KEYS_PER_THREAD

    // Load generator table into shared memory (all threads must participate before syncthreads)
    __shared__ ge25519_p3 s_gen_table[TABLE_SIZE];
    {
        int total_u64s = TABLE_SIZE * 20;
        const u64* src = (const u64*)d_gen_table;
        u64* dst = (u64*)s_gen_table;
        for (int i = threadIdx.x; i < total_u64s; i += blockDim.x) {
            dst[i] = src[i];
        }
        __syncthreads();
    }

    if (tid >= num_threads) return;

    // key_base = tid * KEYS_PER_THREAD
    int key_base = tid << KPT_SHIFT;

    // 1. Load starting point
    ge25519_p3 point;
    for (int i = 0; i < 5; i++) {
        point.X.v[i] = start_X[i];
        point.Y.v[i] = start_Y[i];
        point.Z.v[i] = start_Z[i];
        point.T.v[i] = start_T[i];
    }

    // 2. Add key_base * G using precomputed table
    // Since key_base = tid << KPT_SHIFT, bits 0..(KPT_SHIFT-1) are always 0
    // so we start scanning from bit KPT_SHIFT
    for (int bit = KPT_SHIFT; bit < TABLE_SIZE; bit++) {
        if ((key_base >> bit) & 1) {
            ge25519_p3 tmp;
            ge_add(&tmp, &point, &s_gen_table[bit]);
            point = tmp;
        }
    }

    // 3. Process KEYS_PER_THREAD consecutive keys
    // G = s_gen_table[0] (the generator point)
    for (int k = 0; k < KEYS_PER_THREAD; k++) {
        int key_index = key_base + k;

        // For k > 0, increment point by G
        if (k > 0) {
            ge25519_p3 tmp;
            ge_add(&tmp, &point, &s_gen_table[0]);
            point = tmp;
        }

        // Ristretto compress -> 32 bytes spend_pub
        uint8_t spend_pub[32];
        ristretto_encode(spend_pub, &point);
        uint8_t checksum4[4];
        bool checksum_ready = false;

        // PREFIX CHECK: binary search over sorted, non-overlapping [lo, hi) ranges.
        bool prefix_ok = (num_prefix_ranges == 0);  // no prefix = always match
        if (!prefix_ok) {
            int left = 0;
            int right = num_prefix_ranges;
            while (left < right) {
                int mid = left + ((right - left) >> 1);
                const uint8_t* lo = prefix_ranges + mid * (2 * ADDRESS_TOTAL_BYTES);
                int cmp_lo_sv = cmp_spend_view_split_to_bound(spend_pub, view_pub, lo);
                if (cmp_lo_sv < 0) {
                    right = mid;
                    continue;
                }
                if (cmp_lo_sv == 0) {
                    if (!checksum_ready) {
                        address_checksum4(spend_pub, view_pub, checksum4);
                        checksum_ready = true;
                    }
                    int cmp_lo_checksum = cmp_checksum4_to_bound(checksum4, lo);
                    if (cmp_lo_checksum < 0) {
                        right = mid;
                        continue;
                    }
                }

                const uint8_t* hi = lo + ADDRESS_TOTAL_BYTES;
                int cmp_hi_sv = cmp_spend_view_split_to_bound(spend_pub, view_pub, hi);
                if (cmp_hi_sv < 0) {
                    prefix_ok = true;
                    break;
                }
                if (cmp_hi_sv == 0) {
                    if (!checksum_ready) {
                        address_checksum4(spend_pub, view_pub, checksum4);
                        checksum_ready = true;
                    }
                    int cmp_hi_checksum = cmp_checksum4_to_bound(checksum4, hi);
                    if (cmp_hi_checksum < 0) {
                        prefix_ok = true;
                        break;
                    }
                }

                left = mid + 1;
            }
        }

        // SUFFIX CHECK: modular arithmetic (only if prefix matched)
        bool suffix_ok = (num_suffix_targets == 0);  // no suffix = always match
        if (prefix_ok && !suffix_ok) {
            if (!checksum_ready) {
                address_checksum4(spend_pub, view_pub, checksum4);
                checksum_ready = true;
            }
            u64 mod_val = suffix_mod_for_address(
                spend_pub,
                view_pub,
                checksum4,
                suffix_modulus,
                suffix_chunk_mul,
                suffix_tail_mul
            );

            int t_left = 0;
            int t_right = num_suffix_targets;
            while (t_left < t_right) {
                int t_mid = t_left + ((t_right - t_left) >> 1);
                u64 target = suffix_targets[t_mid];
                if (mod_val < target) {
                    t_right = t_mid;
                } else if (mod_val > target) {
                    t_left = t_mid + 1;
                } else {
                    suffix_ok = true;
                    break;
                }
            }
        }

        out_flags[key_index] = (prefix_ok && suffix_ok) ? 1 : 0;
    }
}

__global__ void __launch_bounds__(256, VANITY_PREFIX_LAUNCH_MIN_BLOCKS) vanity_kernel_prefix(
    const u64* start_X, const u64* start_Y, const u64* start_Z, const u64* start_T,
    const uint8_t* view_pub,
    const uint8_t* prefix_ranges, int num_prefix_ranges,
    int batch_size,
    uint8_t* out_flags
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int num_threads = batch_size >> KPT_SHIFT;

    __shared__ ge25519_p3 s_gen_table[TABLE_SIZE];
    {
        int total_u64s = TABLE_SIZE * 20;
        const u64* src = (const u64*)d_gen_table;
        u64* dst = (u64*)s_gen_table;
        for (int i = threadIdx.x; i < total_u64s; i += blockDim.x) {
            dst[i] = src[i];
        }
        __syncthreads();
    }

    if (tid >= num_threads) return;
    int key_base = tid << KPT_SHIFT;

    ge25519_p3 point;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        point.X.v[i] = start_X[i];
        point.Y.v[i] = start_Y[i];
        point.Z.v[i] = start_Z[i];
        point.T.v[i] = start_T[i];
    }

    for (int bit = KPT_SHIFT; bit < TABLE_SIZE; bit++) {
        if ((key_base >> bit) & 1) {
            ge25519_p3 tmp;
            ge_add(&tmp, &point, &s_gen_table[bit]);
            point = tmp;
        }
    }

    for (int k = 0; k < KEYS_PER_THREAD; k++) {
        int key_index = key_base + k;
        if (k > 0) {
            ge25519_p3 tmp;
            ge_add(&tmp, &point, &s_gen_table[0]);
            point = tmp;
        }

        uint8_t spend_pub[32];
        ristretto_encode(spend_pub, &point);
        uint8_t checksum4[4];
        bool checksum_ready = false;

        bool prefix_ok = false;
        int left = 0;
        int right = num_prefix_ranges;
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            const uint8_t* lo = prefix_ranges + mid * (2 * ADDRESS_TOTAL_BYTES);
            int cmp_lo_sv = cmp_spend_view_split_to_bound(spend_pub, view_pub, lo);
            if (cmp_lo_sv < 0) {
                right = mid;
                continue;
            }
            if (cmp_lo_sv == 0) {
                if (!checksum_ready) {
                    address_checksum4(spend_pub, view_pub, checksum4);
                    checksum_ready = true;
                }
                int cmp_lo_checksum = cmp_checksum4_to_bound(checksum4, lo);
                if (cmp_lo_checksum < 0) {
                    right = mid;
                    continue;
                }
            }

            const uint8_t* hi = lo + ADDRESS_TOTAL_BYTES;
            int cmp_hi_sv = cmp_spend_view_split_to_bound(spend_pub, view_pub, hi);
            if (cmp_hi_sv < 0) {
                prefix_ok = true;
                break;
            }
            if (cmp_hi_sv == 0) {
                if (!checksum_ready) {
                    address_checksum4(spend_pub, view_pub, checksum4);
                    checksum_ready = true;
                }
                int cmp_hi_checksum = cmp_checksum4_to_bound(checksum4, hi);
                if (cmp_hi_checksum < 0) {
                    prefix_ok = true;
                    break;
                }
            }
            left = mid + 1;
        }

        out_flags[key_index] = prefix_ok ? 1 : 0;
    }
}


// ============================================================================
// Host API: Worker lifecycle (persistent memory + per-worker streams)
// ============================================================================

struct CudaWorker {
    // Device memory (persistent)
    u64* d_start_X;
    u64* d_start_Y;
    u64* d_start_Z;
    u64* d_start_T;
    uint8_t* d_view_pub;
    uint8_t view_pub_cache[32];
    int view_pub_initialized;
    char* d_prefix;
    char* d_suffix;
    uint8_t* d_flags;

    // Pinned host memory for flag transfer
    uint8_t* h_flags;

    // Range-based matching (replaces base58 encoding on GPU)
    uint8_t* d_prefix_ranges;    // num_prefix_ranges * 136 bytes (lo[68] || hi[68])
    int num_prefix_ranges;
    u64 suffix_modulus;           // 58^suffix_len, or 0 if no suffix
    u64* d_suffix_targets;        // array of valid mod targets
    int num_suffix_targets;
    u64 suffix_chunk_mul;         // 256^8 mod suffix_modulus
    u64 suffix_tail_mul;          // 256^4 mod suffix_modulus (checksum4 tail)

    // Legacy mode device memory
    uint8_t* d_inputs;

    cudaStream_t stream;
    int max_batch;
    int prefix_len;
    int suffix_len;
    int mode; // 0 = legacy, 1 = full GPU
};

extern "C" int cuda_init_gen_table(const u64* table_data, int num_points) {
    // table_data: num_points * 20 u64s (X[5], Y[5], Z[5], T[5] per point)
    if (num_points > TABLE_SIZE) return 1;

    ge25519_p3 host_table[TABLE_SIZE];
    for (int p = 0; p < num_points; p++) {
        for (int i = 0; i < 5; i++) {
            host_table[p].X.v[i] = table_data[p * 20 + i];
            host_table[p].Y.v[i] = table_data[p * 20 + 5 + i];
            host_table[p].Z.v[i] = table_data[p * 20 + 10 + i];
            host_table[p].T.v[i] = table_data[p * 20 + 15 + i];
        }
    }

    cudaError_t err = cudaMemcpyToSymbol(d_gen_table, host_table,
                                          num_points * sizeof(ge25519_p3));
    if (err != cudaSuccess) return 2;

    return 0;
}

extern "C" void* cuda_worker_create(
    int max_batch,
    const char* prefix, int prefix_len,
    const char* suffix, int suffix_len,
    int mode  // 0=legacy, 1=full GPU keygen
) {
    CudaWorker* w = new CudaWorker();
    w->max_batch = max_batch;
    w->prefix_len = prefix_len;
    w->suffix_len = suffix_len;
    w->mode = mode;

    // Initialize range fields
    w->d_prefix_ranges = nullptr;
    w->num_prefix_ranges = 0;
    w->suffix_modulus = 0;
    w->d_suffix_targets = nullptr;
    w->num_suffix_targets = 0;
    w->suffix_chunk_mul = 0;
    w->suffix_tail_mul = 0;

    cudaStreamCreate(&w->stream);

    // Allocate persistent device memory
    cudaMalloc(&w->d_flags, max_batch);

    // Allocate pinned host memory for fast D→H flag transfer
    cudaMallocHost(&w->h_flags, max_batch);
    cudaMalloc(&w->d_prefix, prefix_len > 0 ? prefix_len : 1);
    cudaMalloc(&w->d_suffix, suffix_len > 0 ? suffix_len : 1);

    // Copy prefix/suffix once (they never change)
    if (prefix_len > 0)
        cudaMemcpy(w->d_prefix, prefix, prefix_len, cudaMemcpyHostToDevice);
    if (suffix_len > 0)
        cudaMemcpy(w->d_suffix, suffix, suffix_len, cudaMemcpyHostToDevice);

    if (mode == 1) {
        // Full GPU mode: allocate point coordinate buffers + view_pub
        cudaMalloc(&w->d_start_X, 5 * sizeof(u64));
        cudaMalloc(&w->d_start_Y, 5 * sizeof(u64));
        cudaMalloc(&w->d_start_Z, 5 * sizeof(u64));
        cudaMalloc(&w->d_start_T, 5 * sizeof(u64));
        cudaMalloc(&w->d_view_pub, 32);
        memset(w->view_pub_cache, 0, sizeof(w->view_pub_cache));
        w->view_pub_initialized = 0;
        w->d_inputs = nullptr;
    } else {
        // Legacy mode: allocate input buffer
        cudaMalloc(&w->d_inputs, (size_t)max_batch * 64);
        w->d_start_X = nullptr;
        w->d_start_Y = nullptr;
        w->d_start_Z = nullptr;
        w->d_start_T = nullptr;
        w->d_view_pub = nullptr;
        memset(w->view_pub_cache, 0, sizeof(w->view_pub_cache));
        w->view_pub_initialized = 0;
    }

    return w;
}

// Set precomputed ranges for range-based matching (replaces base58 on GPU)
extern "C" int cuda_worker_set_ranges(
    void* handle,
    const uint8_t* prefix_ranges, int num_prefix_ranges,  // num_prefix_ranges * 136 bytes
    u64 suffix_modulus,
    const u64* suffix_targets, int num_suffix_targets,
    u64 suffix_chunk_mul,
    u64 suffix_tail_mul
) {
    CudaWorker* w = (CudaWorker*)handle;

    // Free any previous range allocations
    if (w->d_prefix_ranges) cudaFree(w->d_prefix_ranges);
    if (w->d_suffix_targets) cudaFree(w->d_suffix_targets);

    w->num_prefix_ranges = num_prefix_ranges;
    w->suffix_modulus = suffix_modulus;
    w->num_suffix_targets = num_suffix_targets;
    w->suffix_chunk_mul = suffix_chunk_mul;
    w->suffix_tail_mul = suffix_tail_mul;

    // Allocate and copy prefix ranges
    if (num_prefix_ranges > 0) {
        size_t range_size = (size_t)num_prefix_ranges * (2 * ADDRESS_TOTAL_BYTES);
        if (cudaMalloc(&w->d_prefix_ranges, range_size) != cudaSuccess) return 1;
        if (cudaMemcpy(w->d_prefix_ranges, prefix_ranges, range_size, cudaMemcpyHostToDevice) != cudaSuccess) return 2;
    } else {
        w->d_prefix_ranges = nullptr;
    }

    // Allocate and copy suffix targets
    if (num_suffix_targets > 0) {
        size_t targets_size = (size_t)num_suffix_targets * sizeof(u64);
        if (cudaMalloc(&w->d_suffix_targets, targets_size) != cudaSuccess) return 3;
        if (cudaMemcpy(w->d_suffix_targets, suffix_targets, targets_size, cudaMemcpyHostToDevice) != cudaSuccess) return 4;
    } else {
        w->d_suffix_targets = nullptr;
    }

    return 0;
}

// Full GPU mode: submit a batch with starting point coordinates
// Flags are stored in pinned host memory (retrieve with cuda_worker_get_flags)
extern "C" int cuda_worker_submit_v2(
    void* handle,
    const u64* start_X, const u64* start_Y,
    const u64* start_Z, const u64* start_T,
    const uint8_t* view_pub,
    int count
) {
    CudaWorker* w = (CudaWorker*)handle;
    if (count > w->max_batch) return 1;
    if (view_pub == nullptr) return 3;

    cudaStream_t s = w->stream;

    // Copy starting point coordinates (5 u64 each = 40 bytes)
    cudaMemcpyAsync(w->d_start_X, start_X, 5*sizeof(u64), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(w->d_start_Y, start_Y, 5*sizeof(u64), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(w->d_start_Z, start_Z, 5*sizeof(u64), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(w->d_start_T, start_T, 5*sizeof(u64), cudaMemcpyHostToDevice, s);
    // View pub is worker-local and typically constant across all submits.
    // Avoid a redundant tiny H->D copy every batch unless it changed.
    if (!w->view_pub_initialized || memcmp(w->view_pub_cache, view_pub, 32) != 0) {
        memcpy(w->view_pub_cache, view_pub, 32);
        cudaMemcpyAsync(w->d_view_pub, view_pub, 32, cudaMemcpyHostToDevice, s);
        w->view_pub_initialized = 1;
    }

    int threads = 256;
    int num_thread_groups = count / KEYS_PER_THREAD;  // count must be multiple of KEYS_PER_THREAD
    int blocks = (num_thread_groups + threads - 1) / threads;

    if (w->num_prefix_ranges > 0 && w->num_suffix_targets == 0) {
        vanity_kernel_prefix<<<blocks, threads, 0, s>>>(
            w->d_start_X, w->d_start_Y, w->d_start_Z, w->d_start_T,
            w->d_view_pub,
            w->d_prefix_ranges, w->num_prefix_ranges,
            count,
            w->d_flags
        );
    } else if (w->num_prefix_ranges == 0 && w->num_suffix_targets == 0) {
        // No prefix and no suffix: every key matches.
        cudaMemsetAsync(w->d_flags, 1, count, s);
    } else {
        vanity_kernel<<<blocks, threads, 0, s>>>(
            w->d_start_X, w->d_start_Y, w->d_start_Z, w->d_start_T,
            w->d_view_pub,
            w->d_prefix_ranges, w->num_prefix_ranges,
            w->suffix_modulus, w->d_suffix_targets, w->num_suffix_targets,
            w->suffix_chunk_mul, w->suffix_tail_mul,
            count,
            w->d_flags
        );
    }

    // Copy flags to pinned host memory and synchronize
    cudaMemcpyAsync(w->h_flags, w->d_flags, count, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// Get pointer to pinned host flag buffer (valid after submit_v2 returns)
extern "C" const uint8_t* cuda_worker_get_flags(void* handle) {
    CudaWorker* w = (CudaWorker*)handle;
    return w->h_flags;
}

// Legacy mode: submit pre-generated key pairs
extern "C" int cuda_worker_submit(
    void* handle,
    const uint8_t* inputs,
    int count,
    uint8_t* out_flags
) {
    CudaWorker* w = (CudaWorker*)handle;
    if (count > w->max_batch) return 1;

    cudaStream_t s = w->stream;
    size_t input_size = (size_t)count * 64;

    cudaMemcpyAsync(w->d_inputs, inputs, input_size, cudaMemcpyHostToDevice, s);
    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    match_kernel<<<blocks, threads, 0, s>>>(
        w->d_inputs, count,
        w->d_prefix, w->prefix_len,
        w->d_suffix, w->suffix_len,
        w->d_flags
    );

    cudaMemcpyAsync(out_flags, w->d_flags, count, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" void cuda_worker_destroy(void* handle) {
    CudaWorker* w = (CudaWorker*)handle;
    if (!w) return;

    if (w->d_inputs) cudaFree(w->d_inputs);
    if (w->d_start_X) cudaFree(w->d_start_X);
    if (w->d_start_Y) cudaFree(w->d_start_Y);
    if (w->d_start_Z) cudaFree(w->d_start_Z);
    if (w->d_start_T) cudaFree(w->d_start_T);
    if (w->d_view_pub) cudaFree(w->d_view_pub);
    if (w->d_prefix_ranges) cudaFree(w->d_prefix_ranges);
    if (w->d_suffix_targets) cudaFree(w->d_suffix_targets);
    cudaFree(w->d_flags);
    if (w->h_flags) cudaFreeHost(w->h_flags);
    cudaFree(w->d_prefix);
    cudaFree(w->d_suffix);
    cudaStreamDestroy(w->stream);
    delete w;
}

// ============================================================================
// Legacy API (kept for backward compatibility - wraps new worker API)
// ============================================================================

extern "C" int cuda_match_batch(
    const uint8_t* inputs_64, int batch_size,
    const char* prefix_lower, int prefix_len,
    const char* suffix_lower, int suffix_len,
    uint8_t* out_flags
) {
    if (inputs_64 == nullptr || out_flags == nullptr || batch_size <= 0) return 10;

    size_t inputs_size = (size_t)batch_size * 64;
    size_t flags_size = (size_t)batch_size;
    size_t prefix_size = (size_t)(prefix_len > 0 ? prefix_len : 1);
    size_t suffix_size = (size_t)(suffix_len > 0 ? suffix_len : 1);

    uint8_t* d_inputs = nullptr;
    uint8_t* d_flags = nullptr;
    char* d_prefix = nullptr;
    char* d_suffix = nullptr;
    int rc = 0;
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    if (cudaMalloc(&d_inputs, inputs_size) != cudaSuccess) { rc = 11; goto cleanup; }
    if (cudaMalloc(&d_flags, flags_size) != cudaSuccess) { rc = 12; goto cleanup; }
    if (cudaMalloc(&d_prefix, prefix_size) != cudaSuccess) { rc = 13; goto cleanup; }
    if (cudaMalloc(&d_suffix, suffix_size) != cudaSuccess) { rc = 14; goto cleanup; }

    if (cudaMemcpy(d_inputs, inputs_64, inputs_size, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 21; goto cleanup; }
    if (prefix_len > 0 &&
        cudaMemcpy(d_prefix, prefix_lower, (size_t)prefix_len, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 22; goto cleanup; }
    if (suffix_len > 0 &&
        cudaMemcpy(d_suffix, suffix_lower, (size_t)suffix_len, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 23; goto cleanup; }

    match_kernel<<<blocks, threads>>>(d_inputs, batch_size, d_prefix, prefix_len, d_suffix, suffix_len, d_flags);

    if (cudaDeviceSynchronize() != cudaSuccess) { rc = 31; goto cleanup; }
    if (cudaMemcpy(out_flags, d_flags, flags_size, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 32; goto cleanup; }

cleanup:
    if (d_inputs) cudaFree(d_inputs);
    if (d_flags) cudaFree(d_flags);
    if (d_prefix) cudaFree(d_prefix);
    if (d_suffix) cudaFree(d_suffix);
    return rc;
}


// ============================================================================
// Verification kernel: outputs compressed Ristretto bytes for each thread
// Used to validate GPU crypto matches CPU crypto
// ============================================================================

__global__ void verify_compress_kernel(
    const u64* start_X, const u64* start_Y,
    const u64* start_Z, const u64* start_T,
    int count,
    uint8_t* out_compressed  // count * 32 bytes
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    // Load generator table into shared memory (all threads must participate before syncthreads)
    __shared__ ge25519_p3 s_gen_table[TABLE_SIZE];
    {
        int total_u64s = TABLE_SIZE * 20;
        const u64* src = (const u64*)d_gen_table;
        u64* dst = (u64*)s_gen_table;
        for (int i = threadIdx.x; i < total_u64s; i += blockDim.x) {
            dst[i] = src[i];
        }
        __syncthreads();
    }

    if (tid >= count) return;

    // Load starting point
    ge25519_p3 point;
    for (int i = 0; i < 5; i++) {
        point.X.v[i] = start_X[i];
        point.Y.v[i] = start_Y[i];
        point.Z.v[i] = start_Z[i];
        point.T.v[i] = start_T[i];
    }

    // Add tid * G using precomputed table (same as vanity_kernel)
    for (int bit = 0; bit < TABLE_SIZE; bit++) {
        if ((tid >> bit) & 1) {
            ge25519_p3 tmp;
            ge_add(&tmp, &point, &s_gen_table[bit]);
            point = tmp;
        }
    }

    // Ristretto compress and output
    ristretto_encode(&out_compressed[tid * 32], &point);
}

// Diagnostic kernel: outputs intermediate values of ristretto_encode step by step
// Output layout (each 32 bytes):
//   0: u1, 1: u2, 2: inv, 3: den1, 4: den2, 5: z_inv
//   6: ix0, 7: iy0, 8: ench_den, 9: t_zinv
//   10: x (after rotate), 11: y (after rotate), 12: den_inv (after rotate)
//   13: x_zinv, 14: y (after negate), 15: t0 (Z-y), 16: s (before abs), 17: s (final)
//   18: compressed result
// Total: 19 * 32 = 608 bytes
// Plus 1 byte for rotate flag, 1 byte for neg_y flag = 610 bytes
#define DIAG_SIZE 676

__global__ void diag_kernel(
    const u64* start_X, const u64* start_Y,
    const u64* start_Z, const u64* start_T,
    uint8_t* out
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid != 0) return;

    ge25519_p3 point;
    for (int i = 0; i < 5; i++) {
        point.X.v[i] = start_X[i];
        point.Y.v[i] = start_Y[i];
        point.Z.v[i] = start_Z[i];
        point.T.v[i] = start_T[i];
    }

    fe25519 u1, u2, inv, den1, den2, z_inv;
    fe25519 ix0, iy0, ench_den;
    fe25519 x, y, s;

    // u1 = (Z + Y) * (Z - Y)
    fe25519 t0, t1;
    fe_add(&t0, &point.Z, &point.Y);
    fe_sub(&t1, &point.Z, &point.Y);
    fe_mul(&u1, &t0, &t1);

    // u2 = X * Y
    fe_mul(&u2, &point.X, &point.Y);

    // inv = invsqrt(u1 * u2^2)
    fe25519 u2_sq, u1_u2sq, one;
    fe_sq(&u2_sq, &u2);
    fe_mul(&u1_u2sq, &u1, &u2_sq);
    fe_one(&one);
    sqrt_ratio_m1(&inv, &one, &u1_u2sq);

    // den1, den2
    fe_mul(&den1, &inv, &u1);
    fe_mul(&den2, &inv, &u2);

    // z_inv = den1 * den2 * T
    fe_mul(&z_inv, &den1, &den2);
    fe_mul(&z_inv, &z_inv, &point.T);

    fe_mul(&ix0, &point.X, &SQRT_M1);
    fe_mul(&iy0, &point.Y, &SQRT_M1);
    fe_mul(&ench_den, &den1, &INVSQRT_A_MINUS_D);

    fe25519 t_zinv;
    fe_mul(&t_zinv, &point.T, &z_inv);
    int rotate = fe_isneg(&t_zinv);

    fe_copy(&x, &point.X);
    fe_cmov(&x, &iy0, rotate);
    fe_copy(&y, &point.Y);
    fe_cmov(&y, &ix0, rotate);
    fe25519 den_inv;
    fe_copy(&den_inv, &den2);
    fe_cmov(&den_inv, &ench_den, rotate);

    fe25519 x_zinv;
    fe_mul(&x_zinv, &x, &z_inv);
    int neg_y = fe_isneg(&x_zinv);

    fe25519 y_before_neg;
    fe_copy(&y_before_neg, &y);

    fe25519 neg_y_val;
    fe_neg(&neg_y_val, &y);
    fe_cmov(&y, &neg_y_val, neg_y);

    fe_sub(&t0, &point.Z, &y);
    fe_mul(&s, &den_inv, &t0);

    fe25519 s_before_abs;
    fe_copy(&s_before_abs, &s);
    fe_abs(&s, &s);

    // Output all intermediates
    fe_tobytes(&out[0*32], &u1);
    fe_tobytes(&out[1*32], &u2);
    fe_tobytes(&out[2*32], &inv);
    fe_tobytes(&out[3*32], &den1);
    fe_tobytes(&out[4*32], &den2);
    fe_tobytes(&out[5*32], &z_inv);
    fe_tobytes(&out[6*32], &ix0);
    fe_tobytes(&out[7*32], &iy0);
    fe_tobytes(&out[8*32], &ench_den);
    fe_tobytes(&out[9*32], &t_zinv);
    fe_tobytes(&out[10*32], &x);
    fe_tobytes(&out[11*32], &y);
    fe_tobytes(&out[12*32], &den_inv);
    fe_tobytes(&out[13*32], &x_zinv);
    fe_tobytes(&out[14*32], &y_before_neg);
    fe_tobytes(&out[15*32], &t0);      // Z - y
    fe_tobytes(&out[16*32], &s_before_abs);
    fe_tobytes(&out[17*32], &s);
    ristretto_encode(&out[18*32], &point);

    out[608] = (uint8_t)rotate;
    out[609] = (uint8_t)neg_y;
    // Also output sqrt_ratio_m1 return value
    fe_one(&one);
    fe_sq(&u2_sq, &u2);
    fe_mul(&u1_u2sq, &u1, &u2_sq);
    int was_sq = sqrt_ratio_m1(&inv, &one, &u1_u2sq);
    out[610] = (uint8_t)was_sq;
    out[611] = 0;

    // Diagnostic: compare fe_sq(u2) vs fe_mul(u2, u2)
    fe25519 u2_sq_via_sq, u2_sq_via_mul;
    fe_sq(&u2_sq_via_sq, &u2);
    fe_mul(&u2_sq_via_mul, &u2, &u2);
    fe_tobytes(&out[612], &u2_sq_via_sq);
    fe_tobytes(&out[644], &u2_sq_via_mul);
}

extern "C" int cuda_diag_compress(
    const u64* start_X, const u64* start_Y,
    const u64* start_Z, const u64* start_T,
    uint8_t* out  // host buffer: DIAG_SIZE bytes
) {
    u64 *d_X = nullptr, *d_Y = nullptr, *d_Z = nullptr, *d_T = nullptr;
    uint8_t* d_out = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_X, 5*sizeof(u64)) != cudaSuccess) { rc = 1; goto cleanup; }
    if (cudaMalloc(&d_Y, 5*sizeof(u64)) != cudaSuccess) { rc = 2; goto cleanup; }
    if (cudaMalloc(&d_Z, 5*sizeof(u64)) != cudaSuccess) { rc = 3; goto cleanup; }
    if (cudaMalloc(&d_T, 5*sizeof(u64)) != cudaSuccess) { rc = 4; goto cleanup; }
    if (cudaMalloc(&d_out, DIAG_SIZE) != cudaSuccess) { rc = 5; goto cleanup; }

    cudaMemcpy(d_X, start_X, 5*sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, start_Y, 5*sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, start_Z, 5*sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, start_T, 5*sizeof(u64), cudaMemcpyHostToDevice);

    diag_kernel<<<1, 1>>>(d_X, d_Y, d_Z, d_T, d_out);

    if (cudaDeviceSynchronize() != cudaSuccess) { rc = 6; goto cleanup; }
    if (cudaMemcpy(out, d_out, DIAG_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 7; goto cleanup; }

cleanup:
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    if (d_Z) cudaFree(d_Z);
    if (d_T) cudaFree(d_T);
    if (d_out) cudaFree(d_out);
    return rc;
}

extern "C" int cuda_verify_compress(
    const u64* start_X, const u64* start_Y,
    const u64* start_Z, const u64* start_T,
    int count,
    uint8_t* out_compressed  // host buffer: count * 32 bytes
) {
    if (count <= 0 || count > (1 << TABLE_SIZE)) return 1;

    u64 *d_X = nullptr, *d_Y = nullptr, *d_Z = nullptr, *d_T = nullptr;
    uint8_t* d_out = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_X, 5*sizeof(u64)) != cudaSuccess) { rc = 11; goto cleanup; }
    if (cudaMalloc(&d_Y, 5*sizeof(u64)) != cudaSuccess) { rc = 12; goto cleanup; }
    if (cudaMalloc(&d_Z, 5*sizeof(u64)) != cudaSuccess) { rc = 13; goto cleanup; }
    if (cudaMalloc(&d_T, 5*sizeof(u64)) != cudaSuccess) { rc = 14; goto cleanup; }
    if (cudaMalloc(&d_out, (size_t)count * 32) != cudaSuccess) { rc = 15; goto cleanup; }

    cudaMemcpy(d_X, start_X, 5*sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, start_Y, 5*sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, start_Z, 5*sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, start_T, 5*sizeof(u64), cudaMemcpyHostToDevice);

    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        verify_compress_kernel<<<blocks, threads>>>(d_X, d_Y, d_Z, d_T, count, d_out);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) { rc = 31; goto cleanup; }
    if (cudaMemcpy(out_compressed, d_out, (size_t)count * 32, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 32; goto cleanup; }

cleanup:
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    if (d_Z) cudaFree(d_Z);
    if (d_T) cudaFree(d_T);
    if (d_out) cudaFree(d_out);
    return rc;
}
