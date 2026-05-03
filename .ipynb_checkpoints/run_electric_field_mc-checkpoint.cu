/**
 * @file run_electric_field_mc.cu
 * @brief CUDA implementation of the first-stage Electric-Field Monte Carlo (EFMC)
 *        transport used to generate a photon database for later coherent
 *        backscattering processing.
 *
 * High-level purpose
 * ------------------
 * This file runs an Electric Field Monte Carlo of photons interacting with a planetary surface.
 * Many photons are traced in parallel on the GPU. Each GPU thread is responsible
 * for one photon history. The kernel launches, propagates the photon through
 * the layered regolith, samples scattering events from precomputed amplitude
 * scattering tables (S1/S2), updates the local polarization basis, and writes
 * out packet records containing the forward and reversed electric-field data
 * needed by a later post-processing stage.
 *
 * Scope of this file
 * ------------------
 * This file implements the following stage of the workflow:
 *   1. Read precomputed amplitude matrices (from a separate, unincluded code that computes amplitude matrices
 *   from Mueller matrices) and construct scattering phase function tables.
 *   2. Launch GPU transport for many photons.
 *   3. Write a binary photon database (PacketOut records).
 *
 * This file does NOT implement the second-stage database reduction / coherent
 * interference accumulation. That later stage is done in process_photon_packets.c
 *
 * Physics model represented here
 * ------------------------------
 * - Two-layer regolith with layer-specific scattering and absorption inputs.
 * - Scattering events are sampled from tabulated amplitude matrices S1(theta)
 *   and S2(theta).
 * - Polarization is represented as a 2-component complex electric field in the
 *   local (m, n) basis attached to the current propagation direction s.
 * - The code stores enough information to reconstruct forward and time-reversed
 *   paths in a later coherent-backscatter accumulation step.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace {

constexpr int ANGLES = 181;
constexpr float PI = 3.141592653589793238462643383279502884f;
constexpr float RANGE_DEG = 10.0f;
constexpr float ANGLEI_DEG = 0.0f;
constexpr float LAMBDA = 0.126f;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_SCATTER = 200000;

// Physics parameters from the CPU code, with the ZMAX_l1 typo fixed.
constexpr float QXX_l1 = 127.0105178f;
constexpr float SCOEFF_l1 = 1.899049e0f;
constexpr float SSA_l1 = 0.99f;
constexpr float NMED_l1 = 1.6432f;
constexpr float ZMAX_l1 = 18.99049f / SCOEFF_l1;

constexpr float QXX_l2 = 127.0105178f;
constexpr float SCOEFF_l2 = 1.899049e0f;
constexpr float SSA_l2 = 0.99f;
constexpr float NMED_l2 = 1.6432f;
constexpr float ZMAX_l2 = 18.99049f / SCOEFF_l2;

constexpr float ZMAX_total = ZMAX_l1 + ZMAX_l2;

const char *INPUT_L1 = "./Inputs/S1S2_Case3_rock_in_regolith_12_pct.dat";
const char *INPUT_L2 = "./Inputs/S1S2_Case3_rock_in_regolith_12_pct.dat";
const char *OUTPUT_FILE = "photon_db.bin";
constexpr std::uint64_t DEFAULT_RECORDS_PER_PHOTON = 4ULL;

/**
 * @brief Minimal complex-number container used in GPU-safe packet I/O.
 *
 * CUDA can work with C++ complex types in some contexts, but an explicit
 * real/imaginary layout keeps the host/device binary format predictable and
 * easy to read from post-processing tools written in C, C++, Python, or other
 * languages.
 */
struct Complex2 {
    float re;
    float im;
};

/**
 * @brief Lightweight 3-vector used for basis vectors, directions, and positions.
 */
struct Vec3 {
    float x;
    float y;
    float z;
};

/**
 * @brief Binary output record written once a photon contributes an exit event.
 *
 * Each record contains:
 * - Exit angle (`etheta`) in degrees from exact backscatter.
 * - Per-layer path lengths for later absorption weighting.
 * - Exit probability estimate (`pexit`).
 * - Forward/reverse path phase difference used later for interference.
 * - The forward and reversed electric-field vectors.
 * - The local polarization bases for both directions, so the post-processing
 *   stage can rotate the fields into a common frame before interference.
 *
 * This structure is written verbatim to disk. Any reader must use this exact
 * field order and type layout.
 */
struct PacketOut {
    float etheta;
    float pl_l1;
    float pl_l2;
    float pexit;
    float phase;
    int nscatt;
    int _pad;
    Complex2 Efe[2];
    Complex2 Ere[2];
    float mef[3];
    float nef[3];
    float sef[3];
    float mer[3];
    float ner[3];
    float ser[3];
};

/**
 * @brief Host-side container for one layer's amplitude scattering table.
 *
 * `sca_deg` stores the tabulated scattering angles in degrees.
 * `s1` and `s2` are the complex amplitude functions.
 * `cdf` is the cumulative distribution used to sample scattering angle alpha.
 */
struct HostScatteringTable {
    std::vector<float> sca_deg;
    std::vector<Complex2> s1;
    std::vector<Complex2> s2;
    std::vector<float> cdf;
};

/**
 * @brief Collection of raw device pointers to layer-specific scattering tables.
 */
struct DeviceTables {
    Complex2 *s1_l1 = nullptr;
    Complex2 *s2_l1 = nullptr;
    float *cdf_l1 = nullptr;
    Complex2 *s1_l2 = nullptr;
    Complex2 *s2_l2 = nullptr;
    float *cdf_l2 = nullptr;
};

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t err__ = (stmt);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err__), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

/**
 * @brief Construct a Complex2 value.
 */
__host__ __device__ inline Complex2 c_make(float re, float im) {
    Complex2 z{re, im};
    return z;
}

/** @brief Complex addition. */
__host__ __device__ inline Complex2 c_add(Complex2 a, Complex2 b) {
    return c_make(a.re + b.re, a.im + b.im);
}

/** @brief Complex subtraction. */
__host__ __device__ inline Complex2 c_sub(Complex2 a, Complex2 b) {
    return c_make(a.re - b.re, a.im - b.im);
}

/** @brief Complex multiplication. */
__host__ __device__ inline Complex2 c_mul(Complex2 a, Complex2 b) {
    return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

/** @brief Multiply a complex number by a real scalar. */
__host__ __device__ inline Complex2 c_scale(Complex2 a, float s) {
    return c_make(a.re * s, a.im * s);
}

/** @brief Complex conjugate. */
__host__ __device__ inline Complex2 c_conj(Complex2 a) {
    return c_make(a.re, -a.im);
}

/** @brief Squared magnitude |z|^2. */
__host__ __device__ inline float c_abs2(Complex2 a) {
    return a.re * a.re + a.im * a.im;
}

/**
 * @brief Return exp(i * phase).
 *
 * This helper is useful when a phase factor must be applied explicitly to a
 * field. The current first-stage transport stores the phase difference in the
 * packet and leaves the coherent summation to a later stage.
 */
__host__ __device__ inline Complex2 c_exp_i(float phase) {
    return c_make(cosf(phase), sinf(phase));
}

/** @brief Clamp a float to [lo, hi]. */
__host__ __device__ inline float clampd(float x, float lo, float hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

/** @brief Construct a Vec3. */
__host__ __device__ inline Vec3 v_make(float x, float y, float z) {
    Vec3 v{x, y, z};
    return v;
}

/** @brief Euclidean dot product. */
__host__ __device__ inline float dot3(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** @brief Euclidean vector norm. */
__host__ __device__ inline float norm3(const Vec3 &a) {
    return sqrtf(dot3(a, a));
}

/** @brief Scale a vector by a real scalar. */
__host__ __device__ inline Vec3 v_scale(const Vec3 &a, float s) {
    return v_make(a.x * s, a.y * s, a.z * s);
}

/** @brief Vector addition. */
__host__ __device__ inline Vec3 v_add(const Vec3 &a, const Vec3 &b) {
    return v_make(a.x + b.x, a.y + b.y, a.z + b.z);
}

/** @brief Vector subtraction. */
__host__ __device__ inline Vec3 v_sub(const Vec3 &a, const Vec3 &b) {
    return v_make(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Normalize a 3-vector.
 *
 * Returns the zero vector unchanged if the input norm is zero.
 */
__host__ __device__ inline Vec3 normalize3(const Vec3 &a) {
    float n = norm3(a);
    if (n == 0.0f) {
        return v_make(0.0f, 0.0f, 0.0f);
    }
    return v_scale(a, 1.0f / n);
}

/**
 * @brief Multiply two 2x2 complex matrices in row-major.
 *
 * Parameters
 * ----------
 * first, second : input Jones matrices stored as {m00, m01, m10, m11}.
 * product       : output matrix in the same layout.
 *
 * This helper is used repeatedly when building the forward and reversed path
 * operators as in Sawicki et al.
 */
__device__ inline void matrix_multiply_2x2(const Complex2 first[4],
                                           const Complex2 second[4],
                                           Complex2 product[4]) {
    for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
            Complex2 sum = c_make(0.0f, 0.0f);
            for (int k = 0; k < 2; ++k) {
                sum = c_add(sum, c_mul(first[c * 2 + k], second[k * 2 + d]));
            }
            product[c * 2 + d] = sum;
        }
    }
}

/**
 * @brief Rotate the local polarization/propagation frame by (theta, phi).
 *
 * The triplet (m, n, s) is an orthonormal basis where s is the propagation
 * direction and m/n span the transverse polarization plane. These formulas are used after each sampled scatter.
 */
__device__ inline void rotate_ref_frame(Vec3 &m, Vec3 &n, Vec3 &s,
                                        float theta, float phi) {
    Vec3 mrot;
    Vec3 nrot;
    Vec3 srot;

    mrot.x = cosf(theta) * cosf(phi) * m.x + cosf(theta) * sinf(phi) * n.x - sinf(theta) * s.x;
    mrot.y = cosf(theta) * cosf(phi) * m.y + cosf(theta) * sinf(phi) * n.y - sinf(theta) * s.y;
    mrot.z = cosf(theta) * cosf(phi) * m.z + cosf(theta) * sinf(phi) * n.z - sinf(theta) * s.z;

    nrot.x = -sinf(phi) * m.x + cosf(phi) * n.x;
    nrot.y = -sinf(phi) * m.y + cosf(phi) * n.y;
    nrot.z = -sinf(phi) * m.z + cosf(phi) * n.z;

    srot.x = sinf(theta) * cosf(phi) * m.x + sinf(theta) * sinf(phi) * n.x + cosf(theta) * s.x;
    srot.y = sinf(theta) * cosf(phi) * m.y + sinf(theta) * sinf(phi) * n.y + cosf(theta) * s.y;
    srot.z = sinf(theta) * cosf(phi) * m.z + sinf(theta) * sinf(phi) * n.z + cosf(theta) * s.z;

    m = mrot;
    n = nrot;
    s = srot;
}

/**
 * @brief Rotate vector v about axis k by angle theta using Rodrigues' formula.
 */
__device__ inline void rodrigues(const Vec3 &k, float theta, Vec3 &v) {
    float s = sinf(theta);
    float c = cosf(theta);
    float n = 1.0f - c;

    float vx = v.x;
    float vy = v.y;
    float vz = v.z;
    float kx = k.x;
    float ky = k.y;
    float kz = k.z;

    v.x = (kx * kx * n + c) * vx + (ky * kx * n - kz * s) * vy + (kz * kx * n + ky * s) * vz;
    v.y = (kx * ky * n + kz * s) * vx + (ky * ky * n + c) * vy + (ky * kz * n - kx * s) * vz;
    v.z = (kx * kz * n - ky * s) * vx + (ky * kz * n + kx * s) * vy + (kz * kz * n + c) * vz;
}

/**
 * @brief Locate the CDF interval containing a uniform random sample.
 *
 * Returns the lower bin index i such that rr lies in [cdf[i], cdf[i+1]).
 * The later interpolation step converts this into a continuous scattering angle.
 */
__device__ inline int find_alpha_bin(float rr, const float *cdf) {
    int lo = 0;
    int hi = ANGLES - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (rr < cdf[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}

/**
 * @brief Linearly interpolate the tabulated complex amplitudes S1 and S2.
 *
 * @param alpha   Scattering angle in radians.
 * @param s1_tab  Layer-specific S1(theta) table.
 * @param s2_tab  Layer-specific S2(theta) table.
 * @param s1      Output interpolated S1(alpha).
 * @param s2      Output interpolated S2(alpha).
 */
__device__ inline void interp_s1s2(float alpha,
                                   const Complex2 *s1_tab,
                                   const Complex2 *s2_tab,
                                   Complex2 &s1,
                                   Complex2 &s2) {
    float alpha_deg = alpha * 180.0f / PI;
    int i = static_cast<int>(floorf(alpha_deg));
    if (i < 0) i = 0;
    if (i > ANGLES - 2) i = ANGLES - 2;
    float f = alpha_deg - static_cast<float>(i);
    s1 = c_add(s1_tab[i], c_scale(c_sub(s1_tab[i + 1], s1_tab[i]), f));
    s2 = c_add(s2_tab[i], c_scale(c_sub(s2_tab[i + 1], s2_tab[i]), f));
}

/**
 * @brief Select the active layer's optical properties and scattering tables.
 *
 * The current model uses two homogeneous layers stacked along +z. The active
 * layer is determined solely from the photon's current z coordinate.
 */
__device__ inline void select_layer_tables(float z,
                                           const Complex2 *s1_l1,
                                           const Complex2 *s2_l1,
                                           const float *cdf_l1,
                                           const Complex2 *s1_l2,
                                           const Complex2 *s2_l2,
                                           const float *cdf_l2,
                                           const Complex2 *&s1,
                                           const Complex2 *&s2,
                                           const float *&cdf,
                                           float &scoeff,
                                           float &ssa,
                                           float &nmed) {
    if (z < ZMAX_l1) {
        s1 = s1_l1;
        s2 = s2_l1;
        cdf = cdf_l1;
        scoeff = SCOEFF_l1;
        ssa = SSA_l1;
        nmed = NMED_l1;
    } else {
        s1 = s1_l2;
        s2 = s2_l2;
        cdf = cdf_l2;
        scoeff = SCOEFF_l2;
        ssa = SSA_l2;
        nmed = NMED_l2;
    }
}

/**
 * @brief Build the 2x2 Jones scattering matrix for one event.
 *
 * The matrix maps the pre-scatter field components in the old transverse basis
 * into the post-scatter basis defined by the sampled azimuth beta.
 */
__device__ inline void jones_from_s1s2(Complex2 s1, Complex2 s2,
                                       float beta, Complex2 M[4]) {
    float cb = cosf(beta);
    float sb = sinf(beta);
    M[0] = c_scale(s2, cb);
    M[1] = c_scale(s2, sb);
    M[2] = c_scale(s1, -sb);
    M[3] = c_scale(s1, cb);
}

/**
 * @brief Apply a Jones matrix to a 2-component electric field.
 */
__device__ inline void apply_jones(const Complex2 M[4],
                                   const Complex2 Ein[2],
                                   Complex2 Eout[2],
                                   float denom_sqrt) {
    float inv = 1.0f / denom_sqrt;
    Eout[0] = c_scale(c_add(c_mul(M[0], Ein[0]), c_mul(M[1], Ein[1])), inv);
    Eout[1] = c_scale(c_add(c_mul(M[2], Ein[0]), c_mul(M[3], Ein[1])), inv);
}

/**
 * @brief Split a line-of-sight path to the surface into layer-1 and layer-2
 *        optical lengths.
 *
 * This helper is used when reconstructing the forward/reverse phase difference
 * written into the packet database.
 */
__device__ inline void optical_lengths_to_surface(float z, float uz,
                                                  float &L1, float &L2) {
    L1 = 0.0f;
    L2 = 0.0f;
    if (uz == 0.0f) {
        return;
    }

    if (uz > 0.0f) {
        if (z <= ZMAX_l1) {
            L1 = z / uz;
        } else {
            L1 = ZMAX_l1 / uz;
            L2 = (z - ZMAX_l1) / uz;
        }
    } else {
        float up = -uz;
        if (z <= ZMAX_l1) {
            L1 = z / up;
        } else {
            L2 = (z - ZMAX_l1) / up;
            L1 = ZMAX_l1 / up;
        }
    }
}

/**
 * @brief Populate one binary output packet prior to writing it back to host.
 */
__device__ inline void fill_packet(PacketOut &out,
                                   float etheta_deg,
                                   float pl_l1,
                                   float pl_l2,
                                   float pexit,
                                   float phase,
                                   int nscatt,
                                   const Complex2 Efe[2],
                                   const Complex2 Ere[2],
                                   const Vec3 &mef,
                                   const Vec3 &nef,
                                   const Vec3 &sef,
                                   const Vec3 &mer,
                                   const Vec3 &ner,
                                   const Vec3 &ser) {
    out.etheta = etheta_deg;
    out.pl_l1 = pl_l1;
    out.pl_l2 = pl_l2;
    out.pexit = pexit;
    out.phase = phase;
    out.nscatt = nscatt;
    out._pad = 0;
    out.Efe[0] = Efe[0];
    out.Efe[1] = Efe[1];
    out.Ere[0] = Ere[0];
    out.Ere[1] = Ere[1];
    out.mef[0] = mef.x; out.mef[1] = mef.y; out.mef[2] = mef.z;
    out.nef[0] = nef.x; out.nef[1] = nef.y; out.nef[2] = nef.z;
    out.sef[0] = sef.x; out.sef[1] = sef.y; out.sef[2] = sef.z;
    out.mer[0] = mer.x; out.mer[1] = mer.y; out.mer[2] = mer.z;
    out.ner[0] = ner.x; out.ner[1] = ner.y; out.ner[2] = ner.z;
    out.ser[0] = ser.x; out.ser[1] = ser.y; out.ser[2] = ser.z;
}

/**
 * @brief Main GPU transport kernel: one thread traces one photon history.
 *
 * Parameters
 * ----------
 * photon_offset      : Global starting photon index for the current batch.
 * photons_this_batch : Number of photon histories launched in this kernel call.
 * seed               : Base RNG seed used to initialize Philox per thread.
 * s1_l*, s2_l*, cdf_l* : Device scattering tables for the two layers.
 * records            : Output packet buffer in device memory.
 * record_count       : Device-side atomic counter for valid output records.
 * max_records        : Capacity of the output packet buffer for this batch.
 * overflow_flag      : Set to 1 if a batch produces more records than fit in
 *                      the currently allocated packet buffer.
 *
 * Algorithm outline
 * -----------------
 * 1. Initialize the incident field and local basis.
 * 2. Sample an exit direction around exact backscatter.
 * 3. Repeatedly sample a free path, move the photon, and test for escape or
 *    absorption.
 * 4. When a valid exiting contribution is detected, reconstruct the forward and
 *    time-reversed path Jones operators and store one PacketOut record.
 * 5. Continue tracing until escape, absorption, or MAX_SCATTER is reached.
 *
 * Notes
 * -----
 * - This kernel produces the photon database only. It does NOT perform the
 *   later coherent field summation / CBOE binning step.
 */
__global__ void transport_kernel(std::uint64_t photon_offset,
                                 std::uint64_t photons_this_batch,
                                 std::uint64_t seed,
                                 const Complex2 *s1_l1,
                                 const Complex2 *s2_l1,
                                 const float *cdf_l1,
                                 const Complex2 *s1_l2,
                                 const Complex2 *s2_l2,
                                 const float *cdf_l2,
                                 PacketOut *records,
                                 unsigned int *record_count,
                                 unsigned int max_records,
                                 unsigned int *overflow_flag) {
    std::uint64_t local_id = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_id >= photons_this_batch) {
        return;
    }

    std::uint64_t global_id = photon_offset + local_id;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, global_id, 0ULL, &rng);

    // Incident frame attached to the incoming wave before any scattering.
    Vec3 mi = v_make(1.0f, 0.0f, 0.0f);
    Vec3 ni = v_make(0.0f, 1.0f, 0.0f);
    Vec3 si = v_make(0.0f, 0.0f, 1.0f);

    float itheta = ANGLEI_DEG * PI / 180.0f;
    float iphi = 0.0f;
    rotate_ref_frame(mi, ni, si, itheta, iphi);

    Complex2 Ei[2];
    Ei[0] = c_make(1.0f / sqrtf(2.0f), 0.0f);
    Ei[1] = c_make(0.0f, -1.0f / sqrtf(2.0f));

    Complex2 E[2] = {Ei[0], Ei[1]};
    Vec3 m = mi;
    Vec3 n = ni;
    Vec3 s = si;

    Complex2 Mfwd[4];
    Complex2 Mprev[4];
    Complex2 Mf0[4];
    for (int i = 0; i < 4; ++i) {
        Mfwd[i] = c_make(0.0f, 0.0f);
        Mprev[i] = c_make(0.0f, 0.0f);
        Mf0[i] = c_make(0.0f, 0.0f);
    }
    Mfwd[0] = c_make(1.0f, 0.0f);
    Mfwd[3] = c_make(1.0f, 0.0f);
    Mprev[0] = c_make(1.0f, 0.0f);
    Mprev[3] = c_make(1.0f, 0.0f);

    float rr = curand_uniform(&rng);
    float etheta = PI - rr * RANGE_DEG * PI / 180.0f;

    Vec3 me = mi;
    Vec3 ne = ni;
    Vec3 se = si;
    Vec3 se0 = v_scale(si, -1.0f);
    rotate_ref_frame(me, ne, se, etheta, 0.0f);

    rr = curand_uniform(&rng);
    float ephi = 2.0f * PI * rr;
    // Rotate about the backscatter axis se0.
    rodrigues(se0, ephi, me);
    rodrigues(se0, ephi, ne);
    rodrigues(se0, ephi, se);
    float ephi1 = atan2f(se.y, se.x);

    float dx = si.x;
    float dy = si.y;
    float dz = si.z;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    int nscatt = 0;
    float pl_l1 = 0.0f;
    float pl_l2 = 0.0f;
    float pbeta_prev = 1.0f;

    Vec3 mold = mi;
    Vec3 nold = ni;
    Vec3 sold = si;
    Complex2 Eold[2] = {Ei[0], Ei[1]};

    Vec3 m1 = mi;
    Vec3 n1 = ni;
    Vec3 s1 = si;
    float x1 = 0.0f, y1 = 0.0f, z1 = 0.0f;

    for (int scatter_iter = 0; scatter_iter < MAX_SCATTER; ++scatter_iter) {
        const Complex2 *s1_tab = nullptr;
        const Complex2 *s2_tab = nullptr;
        const float *cdf = nullptr;
        float scoeff = 0.0f;
        float ssa = 0.0f;
        float nmed = 0.0f;
        select_layer_tables(z, s1_l1, s2_l1, cdf_l1, s1_l2, s2_l2, cdf_l2,
                            s1_tab, s2_tab, cdf, scoeff, ssa, nmed);

        // Exponential free-path sampling with layer-specific scattering
        // coefficient. Units are meters because scoeff is in 1/m.
        rr = curand_uniform(&rng);
        rr = (rr == 0.0f) ? 1.0e-16f : rr;
        float dl = logf(1.0f / rr) / scoeff;

        // Before applying the move, check whether the sampled free path would
        // carry the photon out of the top surface. If so, compute one database
        // record representing that escape contribution.
        if (nscatt > 0 && z + dl * se.z < 0.0f) {
            float dxrot = dot3(se, mold);
            float dyrot = dot3(se, nold);
            float dzrot = dot3(se, sold);
            float beta = atan2f(dyrot, dxrot);
            float alpha = acosf(clampd(dzrot, -1.0f, 1.0f));

            Vec3 mef = mold;
            Vec3 nef = nold;
            Vec3 sef = sold;
            rotate_ref_frame(mef, nef, sef, alpha, beta);
            Vec3 rot_axis = v_make(sinf(ephi1), -cosf(ephi1), 0.0f);
            rodrigues(rot_axis, 0.0f, mef);
            rodrigues(rot_axis, 0.0f, nef);
            rodrigues(rot_axis, 0.0f, sef);

            Complex2 iS1, iS2;
            interp_s1s2(alpha, s1_tab, s2_tab, iS1, iS2);
            float S11 = 0.5f * (c_abs2(iS2) + c_abs2(iS1));
            float S12 = 0.5f * (c_abs2(iS2) - c_abs2(iS1));
            float SI = c_abs2(Eold[0]) + c_abs2(Eold[1]);
            float SQ = c_abs2(Eold[0]) - c_abs2(Eold[1]);
            float SU = 2.0f * (Eold[0].re * Eold[1].re + Eold[0].im * Eold[1].im);
            float pexit = S11 * SI + S12 * (SQ * cosf(2.0f * beta) + SU * sinf(2.0f * beta));

            Complex2 Mfn[4];
            jones_from_s1s2(iS1, iS2, beta, Mfn);

            if (nscatt > 1) {
                float xn = x;
                float yn = y;
                float zn = z;

                Complex2 Q[4] = {
                    c_make(1.0f, 0.0f), c_make(0.0f, 0.0f),
                    c_make(0.0f, 0.0f), c_make(-1.0f, 0.0f)
                };
                Complex2 MfwdT[4] = {Mfwd[0], Mfwd[2], Mfwd[1], Mfwd[3]};
                Complex2 Mp[4], Mrev[4], Mpp[4];
                matrix_multiply_2x2(MfwdT, Q, Mp);
                matrix_multiply_2x2(Q, Mp, Mrev);

                dxrot = -dot3(sold, mi);
                dyrot = -dot3(sold, ni);
                dzrot = -dot3(sold, si);
                beta = atan2f(dyrot, dxrot);
                alpha = acosf(clampd(dzrot, -1.0f, 1.0f));

                Vec3 mrot = mi;
                Vec3 nrot = ni;
                Vec3 srot = si;
                rotate_ref_frame(mrot, nrot, srot, alpha, beta);
                interp_s1s2(alpha, s1_tab, s2_tab, iS1, iS2);
                Complex2 Mrn[4];
                jones_from_s1s2(iS1, iS2, beta, Mrn);

                float cos_beta1 = clampd(dot3(mold, mrot), -1.0f, 1.0f);
                float beta1 = acosf(cos_beta1);
                if (acosf(clampd(dot3(mold, nrot), -1.0f, 1.0f)) > 0.5f * PI) {
                    beta1 = 2.0f * PI - beta1;
                }
                Complex2 Rbeta1[4] = {
                    c_make(cosf(beta1), 0.0f), c_make(sinf(beta1), 0.0f),
                    c_make(-sinf(beta1), 0.0f), c_make(cosf(beta1), 0.0f)
                };

                dxrot = dot3(se, m1);
                dyrot = -dot3(se, n1);
                dzrot = -dot3(se, s1);
                beta = atan2f(dyrot, dxrot);
                alpha = acosf(clampd(dzrot, -1.0f, 1.0f));

                Vec3 mer = m1;
                Vec3 ner = v_scale(n1, -1.0f);
                Vec3 ser = v_scale(s1, -1.0f);
                rotate_ref_frame(mer, ner, ser, alpha, beta);
                rodrigues(rot_axis, 0.0f, mer);
                rodrigues(rot_axis, 0.0f, ner);
                rodrigues(rot_axis, 0.0f, ser);

                const Complex2 *s1_first = nullptr;
                const Complex2 *s2_first = nullptr;
                const float *cdf_first = nullptr;
                float scoeff_first = 0.0f;
                float ssa_first = 0.0f;
                float nmed_first = 0.0f;
                select_layer_tables(z1, s1_l1, s2_l1, cdf_l1, s1_l2, s2_l2, cdf_l2,
                                    s1_first, s2_first, cdf_first,
                                    scoeff_first, ssa_first, nmed_first);
                interp_s1s2(alpha, s1_first, s2_first, iS1, iS2);
                Complex2 Mr0[4];
                jones_from_s1s2(iS1, iS2, beta, Mr0);

                Complex2 Mre[4], Mfe[4];
                matrix_multiply_2x2(Rbeta1, Mrn, Mp);
                matrix_multiply_2x2(Mrev, Mp, Mpp);
                matrix_multiply_2x2(Mr0, Mpp, Mre);
                matrix_multiply_2x2(Mfwd, Mf0, Mp);
                matrix_multiply_2x2(Mfn, Mp, Mfe);

                Complex2 Efe[2], Ere[2];
                Efe[0] = c_add(c_mul(Mfe[0], Ei[0]), c_mul(Mfe[1], Ei[1]));
                Efe[1] = c_add(c_mul(Mfe[2], Ei[0]), c_mul(Mfe[3], Ei[1]));
                Ere[0] = c_add(c_mul(Mre[0], Ei[0]), c_mul(Mre[1], Ei[1]));
                Ere[1] = c_add(c_mul(Mre[2], Ei[0]), c_mul(Mre[3], Ei[1]));

                float l1si_1, l1si_2;
                float lnse_1, lnse_2;
                float lnsi_1, lnsi_2;
                float l1se_1, l1se_2;
                optical_lengths_to_surface(z1, si.z, l1si_1, l1si_2);
                optical_lengths_to_surface(zn, se.z, lnse_1, lnse_2);
                optical_lengths_to_surface(zn, si.z, lnsi_1, lnsi_2);
                optical_lengths_to_surface(z1, se.z, l1se_1, l1se_2);

                float lnse = (zn / -se.z);
                float xef = xn + se.x * lnse;
                float yef = yn + se.y * lnse;
                float zef = zn + se.z * lnse;
                float lnsi = zn / si.z;
                float xir = xn - si.x * lnsi;
                float yir = yn - si.y * lnsi;
                float zir = zn - si.z * lnsi;
                float l1se = -z1 / se.z;
                float xer = x1 + se.x * l1se;
                float yer = y1 + se.y * l1se;
                float zer = z1 + se.z * l1se;
                float d1 = sef.x * (xer - xef) + sef.y * (yer - yef) + sef.z * (zer - zef);
                float d2 = si.x * xir + si.y * yir + si.z * zir;

                float optical_diff = NMED_l1 * (l1si_1 + lnse_1 - lnsi_1 - l1se_1)
                                    + NMED_l2 * (l1si_2 + lnse_2 - lnsi_2 - l1se_2);
                float phase = (2.0f * PI / LAMBDA) * (optical_diff + d1 - d2);

                // Reserve a record slot atomically because many GPU threads
                // can discover valid escape contributions concurrently.
                unsigned int slot = atomicAdd(record_count, 1U);
                if (slot < max_records) {
                    fill_packet(records[slot],
                                (PI - etheta) * 180.0f / PI,
                                pl_l1,
                                pl_l2,
                                pexit,
                                phase,
                                nscatt,
                                Efe,
                                Ere,
                                mef,
                                nef,
                                sef,
                                mer,
                                ner,
                                ser);
                } else {
                    atomicExch(overflow_flag, 1U);
                }
            } else {
                Complex2 Eout[2];
                Eout[0] = c_add(c_mul(Mfn[0], Ei[0]), c_mul(Mfn[1], Ei[1]));
                Eout[1] = c_add(c_mul(Mfn[2], Ei[0]), c_mul(Mfn[3], Ei[1]));
                unsigned int slot = atomicAdd(record_count, 1U);
                if (slot < max_records) {
                    fill_packet(records[slot],
                                (PI - etheta) * 180.0f / PI,
                                pl_l1,
                                pl_l2,
                                pexit,
                                0.0f,
                                nscatt,
                                Eout,
                                Eout,
                                mef,
                                nef,
                                sef,
                                mef,
                                nef,
                                sef);
                } else {
                    atomicExch(overflow_flag, 1U);
                }
            }
        }

        // Apply the sampled move now that any top-surface contribution has
        // been recorded.
        float xnew = x + dl * dx;
        float ynew = y + dl * dy;
        float znew = z + dl * dz;

        if (z < ZMAX_l1) {
            pl_l1 += dl;
        } else {
            pl_l2 += dl;
        }

        x = xnew;
        y = ynew;
        z = znew;

        if (z < 0.0f || z > ZMAX_total) {
            break;
        }

        rr = curand_uniform(&rng);
        if (rr > ssa) {
            break;
        }

        if (nscatt > 1) {
            Complex2 Mp[4];
            matrix_multiply_2x2(Mprev, Mfwd, Mp);
            float inv = 1.0f / sqrtf(fmaxf(pbeta_prev, 1.0e-30f));
            for (int i = 0; i < 4; ++i) {
                Mfwd[i] = c_scale(Mp[i], inv);
            }
        }

        // Sample the next scattering deflection alpha from the precomputed
        // CDF for the active layer.
        rr = curand_uniform(&rng);
        int ibin = find_alpha_bin(rr, cdf);
        float denom = cdf[ibin + 1] - cdf[ibin];
        float frac = (denom > 0.0f) ? (rr - cdf[ibin]) / denom : 0.0f;
        frac = clampd(frac, 0.0f, 1.0f);
        float alpha = (static_cast<float>(ibin) + frac) * PI / 180.0f;

        Complex2 iS1, iS2;
        interp_s1s2(alpha, s1_tab, s2_tab, iS1, iS2);
        float S11 = 0.5f * (c_abs2(iS2) + c_abs2(iS1));
        float S12 = 0.5f * (c_abs2(iS2) - c_abs2(iS1));
        float SI = c_abs2(E[0]) + c_abs2(E[1]);
        float SQ = c_abs2(E[0]) - c_abs2(E[1]);
        float SU = 2.0f * (E[0].re * E[1].re + E[0].im * E[1].im);
        float ptheta = 2.0f * PI * S11;

        // Sample the azimuth beta using rejection sampling conditioned on the
        // current polarization state.
        float pcond = -1.0f;
        float beta = 0.0f;
        float pbeta = 1.0f;
        while (pcond < curand_uniform(&rng)) {
            beta = 2.0f * PI * curand_uniform(&rng);
            pbeta = S11 * SI + S12 * (SQ * cosf(2.0f * beta) + SU * sinf(2.0f * beta));
            pcond = pbeta / ptheta;
        }
        pbeta_prev = pbeta;

        mold = m;
        nold = n;
        sold = s;
        Eold[0] = E[0];
        Eold[1] = E[1];
        rotate_ref_frame(m, n, s, alpha, beta);
        dx = s.x;
        dy = s.y;
        dz = s.z;

        ++nscatt;

        // Build and apply the Jones operator for this event, then rotate the
        // propagation frame to the new direction.
        jones_from_s1s2(iS1, iS2, beta, Mprev);
        Complex2 Enew[2];
        apply_jones(Mprev, E, Enew, sqrtf(fmaxf(pbeta, 1.0e-30f)));
        E[0] = Enew[0];
        E[1] = Enew[1];

        if (nscatt == 1) {
            for (int i = 0; i < 4; ++i) {
                Mf0[i] = Mprev[i];
            }
            m1 = m;
            n1 = n;
            s1 = s;
            x1 = x;
            y1 = y;
            z1 = z;
        }
    }

}

/**
 * @brief Read one amplitude-scattering file and build its sampling CDF.
 *
 * The input text file is expected to contain one row per degree with columns
 *   angle_deg  Re(S1)  Im(S1)  Re(S2)  Im(S2)
 * After reading, the amplitudes are normalized by sqrtf(pi * Qxx) and the CDF
 * used for alpha sampling is constructed from the tabulated differential power.
 */
HostScatteringTable load_scattering_table(const char *path, float qxx) {
    HostScatteringTable table;
    table.sca_deg.resize(ANGLES);
    table.s1.resize(ANGLES);
    table.s2.resize(ANGLES);
    table.cdf.resize(ANGLES);

    FILE *fp = std::fopen(path, "r");
    if (!fp) {
        throw std::runtime_error(std::string("Failed to open scattering file: ") + path);
    }

    float sum = 0.0f;
    for (int i = 0; i < ANGLES; ++i) {
        float reS1, imS1, reS2, imS2;
        if (std::fscanf(fp, "%f %f %f %f %f", &table.sca_deg[i], &reS1, &imS1, &reS2, &imS2) != 5) {
            std::fclose(fp);
            throw std::runtime_error(std::string("Malformed scattering file: ") + path);
        }
        float norm = sqrtf(PI * qxx);
        table.s1[i] = c_make(reS1 / norm, imS1 / norm);
        table.s2[i] = c_make(reS2 / norm, imS2 / norm);
        float ptheta = PI * (c_abs2(table.s2[i]) + c_abs2(table.s1[i]));
        sum += ptheta * sinf(static_cast<float>(i) * PI / 180.0f);
        table.cdf[i] = sum;
    }
    std::fclose(fp);

    if (sum <= 0.0f) {
        throw std::runtime_error(std::string("Degenerate scattering CDF for file: ") + path);
    }
    for (int i = 0; i < ANGLES; ++i) {
        table.cdf[i] /= sum;
    }
    table.cdf[ANGLES - 1] = 1.0f;
    return table;
}

/**
 * @brief Allocate GPU memory for the scattering tables and copy host data to it.
 */
void copy_tables_to_device(const HostScatteringTable &l1,
                           const HostScatteringTable &l2,
                           DeviceTables &dt) {
    CUDA_CHECK(cudaMalloc(&dt.s1_l1, ANGLES * sizeof(Complex2)));
    CUDA_CHECK(cudaMalloc(&dt.s2_l1, ANGLES * sizeof(Complex2)));
    CUDA_CHECK(cudaMalloc(&dt.cdf_l1, ANGLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt.s1_l2, ANGLES * sizeof(Complex2)));
    CUDA_CHECK(cudaMalloc(&dt.s2_l2, ANGLES * sizeof(Complex2)));
    CUDA_CHECK(cudaMalloc(&dt.cdf_l2, ANGLES * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dt.s1_l1, l1.s1.data(), ANGLES * sizeof(Complex2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.s2_l1, l1.s2.data(), ANGLES * sizeof(Complex2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.cdf_l1, l1.cdf.data(), ANGLES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.s1_l2, l2.s1.data(), ANGLES * sizeof(Complex2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.s2_l2, l2.s2.data(), ANGLES * sizeof(Complex2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.cdf_l2, l2.cdf.data(), ANGLES * sizeof(float), cudaMemcpyHostToDevice));
}

/** @brief Release all device table allocations. */
void free_device_tables(DeviceTables &dt) {
    cudaFree(dt.s1_l1);
    cudaFree(dt.s2_l1);
    cudaFree(dt.cdf_l1);
    cudaFree(dt.s1_l2);
    cudaFree(dt.s2_l2);
    cudaFree(dt.cdf_l2);
    dt = DeviceTables{};
}

/**
 * @brief Parse an unsigned 64-bit integer from the command line or return a
 *        caller-supplied default.
 */
std::uint64_t parse_u64_or_default(const char *arg, std::uint64_t default_value) {
    if (!arg) {
        return default_value;
    }
    char *end = nullptr;
    unsigned long long v = std::strtoull(arg, &end, 10);
    return (end && *end == '\0') ? static_cast<std::uint64_t>(v) : default_value;
}

} // namespace

/**
 * @brief Host entry point.
 *
 * Command-line arguments
 * ----------------------
 * argv[1] : total number of photons to launch
 * argv[2] : batch size (how many photons to process per kernel launch)
 * argv[3] : RNG seed
 * argv[4] : packet-capacity multiplier (records allocated per photon in each
 *           batch, default = 4)
 *
 * Workflow
 * --------
 * 1. Load layer-specific scattering tables from disk.
 * 2. Copy those tables to the GPU.
 * 3. Allocate an output packet buffer on the device that is larger than the
 *    photon batch size, using a configurable records-per-photon multiplier (cause each photon can add more than 
 *    one record to the output).
 * 4. Launch the transport kernel repeatedly until the requested photon count is
 *    exhausted.
 * 5. Copy valid packet records back to the host and append them to the output
 *    binary database file.
 */
int main(int argc, char **argv) {
    try {
        std::uint64_t total_photons = (argc > 1) ? parse_u64_or_default(argv[1], 50000000ULL) : 50000000ULL;
        std::uint64_t batch_photons = (argc > 2) ? parse_u64_or_default(argv[2], 1000000ULL) : 1000000ULL;
        std::uint64_t seed = (argc > 3) ? parse_u64_or_default(argv[3], 123456789ULL) : 123456789ULL;
        std::uint64_t records_per_photon = (argc > 4) ? parse_u64_or_default(argv[4], DEFAULT_RECORDS_PER_PHOTON) : DEFAULT_RECORDS_PER_PHOTON;

        if (records_per_photon < 3ULL) {
            throw std::runtime_error("records_per_photon must be at least 3 so the packet buffer is larger than the photon batch and there are no issues with memory allocation.");
        }

        HostScatteringTable host_l1 = load_scattering_table(INPUT_L1, QXX_l1);
        HostScatteringTable host_l2 = load_scattering_table(INPUT_L2, QXX_l2);

        DeviceTables dt;
        copy_tables_to_device(host_l1, host_l2, dt);

        if (batch_photons > std::numeric_limits<std::uint64_t>::max() / records_per_photon) {
            throw std::runtime_error("batch_photons * records_per_photon overflowed 64-bit capacity computation.");
        }
        std::uint64_t record_capacity = batch_photons * records_per_photon;
        if (record_capacity > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::runtime_error("record_capacity exceeds the 32-bit device counter range; reduce batch size or records_per_photon.");
        }

        PacketOut *d_records = nullptr;
        unsigned int *d_record_count = nullptr;
        unsigned int *d_overflow_flag = nullptr;
        CUDA_CHECK(cudaMalloc(&d_records, static_cast<size_t>(record_capacity) * sizeof(PacketOut)));
        CUDA_CHECK(cudaMalloc(&d_record_count, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_overflow_flag, sizeof(unsigned int)));

        std::vector<PacketOut> h_records(static_cast<size_t>(record_capacity));
        FILE *fp = std::fopen(OUTPUT_FILE, "wb");
        if (!fp) {
            throw std::runtime_error(std::string("Failed to open output file: ") + OUTPUT_FILE);
        }

        for (std::uint64_t photon_offset = 0; photon_offset < total_photons; photon_offset += batch_photons) {
            std::uint64_t photons_this_batch = std::min(batch_photons, total_photons - photon_offset);
            unsigned int zero = 0U;
            CUDA_CHECK(cudaMemcpy(d_record_count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_overflow_flag, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

            int blocks = static_cast<int>((photons_this_batch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            transport_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                photon_offset,
                photons_this_batch,
                seed,
                dt.s1_l1,
                dt.s2_l1,
                dt.cdf_l1,
                dt.s1_l2,
                dt.s2_l2,
                dt.cdf_l2,
                d_records,
                d_record_count,
                static_cast<unsigned int>(record_capacity),
                d_overflow_flag);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned int produced = 0U;
            unsigned int overflowed = 0U;
            CUDA_CHECK(cudaMemcpy(&produced, d_record_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&overflowed, d_overflow_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (overflowed != 0U || produced > static_cast<unsigned int>(record_capacity)) {
                throw std::runtime_error("Packet buffer overflow: increase records_per_photon or reduce batch_photons.");
            }
            unsigned int stored = produced;
            if (stored > 0U) {
                CUDA_CHECK(cudaMemcpy(h_records.data(), d_records,
                                      static_cast<size_t>(stored) * sizeof(PacketOut), cudaMemcpyDeviceToHost));
                size_t written = std::fwrite(h_records.data(), sizeof(PacketOut), stored, fp);
                if (written != stored) {
                    throw std::runtime_error("Failed while writing output records.");
                }
            }
            std::fprintf(stdout,
                         "Processed photons [%llu, %llu) -> %u records stored (capacity %llu)\n",
                         static_cast<unsigned long long>(photon_offset),
                         static_cast<unsigned long long>(photon_offset + photons_this_batch),
                         stored,
                         static_cast<unsigned long long>(record_capacity));
        }

        std::fclose(fp);
        CUDA_CHECK(cudaFree(d_records));
        CUDA_CHECK(cudaFree(d_record_count));
        CUDA_CHECK(cudaFree(d_overflow_flag));
        free_device_tables(dt);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "Fatal error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}

