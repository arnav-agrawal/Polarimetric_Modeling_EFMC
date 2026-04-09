/**
 * @file output_photon_packet_structure.h
 * @brief Shared binary packet definition for EFMC transport output.
 *
 * Both the GPU transport code and any later post-processing code must include
 * this header so they agree on the exact byte layout written to disk.
 */

#ifndef EFMC_PACKET_LAYOUT_H
#define EFMC_PACKET_LAYOUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Plain real/imaginary complex-number storage used in packets. */
typedef struct {
    double re;
    double im;
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
 */
typedef struct {
    double etheta;
    double pl_l1;
    double pl_l2;
    double pexit;
    double phase;
    int nscatt;
    int _pad;
    Complex2 Efe[2];
    Complex2 Ere[2];
    double mef[3];
    double nef[3];
    double sef[3];
    double mer[3];
    double ner[3];
    double ser[3];
} PacketOut;

#ifdef __cplusplus
}
#endif

#endif
