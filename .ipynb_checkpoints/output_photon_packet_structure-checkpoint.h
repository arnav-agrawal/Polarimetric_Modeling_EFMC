/**
 * @file output_photon_packet_structure.h
 * @brief Shared binary packet definition for EFMC transport output.
 *
 * Both the GPU transport code and any later post-processing code must include
 * this header so they agree on the exact byte layout written to disk.
 *
 */

#ifndef EFMC_PACKET_LAYOUT_H
#define EFMC_PACKET_LAYOUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Plain real/imaginary complex-number storage used in packets. */
typedef struct {
    float re;
    float im;
} Complex2;

/**
 * @brief One photon database record emitted by the Monte Carlo stage.
 *
 * Fields
 * ------
 * etheta : exit angle in degrees relative to exact backscatter
 * pl_l1  : path length traveled in layer 1 [m]
 * pl_l2  : path length traveled in layer 2 [m]
 * pexit  : probability-like exit weighting
 * phase  : forward/reverse phase difference used later for coherent summation
 * nscatt : number of scattering events in the path
 * Efe    : forward electric field (2 transverse components)
 * Ere    : reverse-path electric field (2 transverse components)
 * mef...ser : basis vectors needed to rotate the fields into a common frame
 *
 * Binary layout note
 * ------------------
 * This structure is written verbatim by run_electric_field_mc_float.cu.
 * Do not change field order, field type, or padding unless the writer is
 * changed at the same time.
 */
typedef struct {
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
} PacketOut;

#ifdef __cplusplus
}
#endif

/* The CUDA float writer emits 132-byte PacketOut records on the target layout. */
#if defined(__cplusplus)
static_assert(sizeof(Complex2) == 2 * sizeof(float), "Complex2 layout mismatch");
static_assert(sizeof(PacketOut) == 132, "PacketOut float layout mismatch");
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(Complex2) == 2 * sizeof(float), "Complex2 layout mismatch");
_Static_assert(sizeof(PacketOut) == 132, "PacketOut float layout mismatch");
#endif

#endif
