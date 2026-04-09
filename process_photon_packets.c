#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include "output_photon_packet_structure.h"

#define RANGE 10.0     /* in degrees */
#define NBINS 100      /* number of bins in range */
#define SMALL 1.0E-20

#define ACOEFF_l1 9.104343E-02 /* absorption coefficient for layer 1 [per m] */
#define ACOEFF_l2 9.104343E-02 /* absorption coefficient for layer 2 [per m] */

/*
 * This reducer is adapted to the CUDA packet layout defined in
 * output_photon_packet_structure.h. It computes the necessary math from the output packets to calculute 
 * the coherent summation of backscattered photons.
 */

/* Compute product = first x second */
static void matrix_multiply(const double first[16], const double second[16], double product[16])
{
    int c, d, k;
    double sum;

    for (c = 0; c < 4; c++)
    {
        for (d = 0; d < 4; d++)
        {
            sum = 0.0;
            for (k = 0; k < 4; k++)
            {
                sum += first[c * 4 + k] * second[k * 4 + d];
            }
            product[c * 4 + d] = sum;
        }
    }
}

/* Return dot product a.b */
static double dot3_arr(const double a[3], const double b[3])
{
    int i;
    double dot_product = 0.0;

    for (i = 0; i < 3; i++)
    {
        dot_product += a[i] * b[i];
    }

    return dot_product;
}

/* Convert explicit packet complex storage to C double complex. */
static inline double complex c_from_packet(Complex2 z)
{
    return z.re + I * z.im;
}

int main(int argc, char **argv)
{
    int i, j, k, ibin, nsteps;
    double dtheta, etheta, pl_l1, pl_l2, pexit, phase, weight, ncount, ep, phi;
    double If, Qf, Uf, Vf, Ir, Qr, Ur, Vr;
    double count_ms[NBINS], count_ss[NBINS], SI[NBINS], SQ[NBINS], SU[NBINS], SV[NBINS];
    double incI[NBINS], incQ[NBINS], incU[NBINS], incV[NBINS];
    double reEf0, imEf0, reEf1, imEf1, reEr0, imEr0, reEr1, imEr1;
    double mef[3], nef[3], sef[3], mer[3], ner[3], ser[3];
    double mrotf[3], nrotf[3], srotf[3], mrotr[3], nrotr[3], srotr[3];
    double mrotf2[3], nrotf2[3], srotf2[3], mrotr2[3], nrotr2[3], srotr2[3];
    double complex Efwd[2], Erev[2], Efrot[2], Errot[2], E0, E1;
    double Sfe[4], Sre[4], Rep[16], Rphi[16], Mrot[16], Sfrot[4], Srrot[4];
    double polf, polr;

    const char *input_path = (argc > 1) ? argv[1] : "photon_db.bin";
    const char *output_path = (argc > 2) ? argv[2] : "./Outputs/test_run_for_code_review_share.dat";

    FILE *fp;
    PacketOut op;
    unsigned long long records_read = 0ULL;
    const unsigned long long progress_interval = 1000000ULL; /* print every 1M packet records */

    dtheta = RANGE / NBINS;

    for (i = 0; i < NBINS; i++)
    {
        SI[i] = 0.0;
        SQ[i] = 0.0;
        SU[i] = 0.0;
        SV[i] = 0.0;

        incI[i] = 0.0;
        incQ[i] = 0.0;
        incU[i] = 0.0;
        incV[i] = 0.0;

        count_ms[i] = 0.0;
        count_ss[i] = 0.0;
    }

    fp = fopen(input_path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open input file: %s\n", input_path);
        return EXIT_FAILURE;
    }

    printf("Reading packet database: %s\n", input_path);
    printf("Writing reduced output to: %s\n", output_path);
    printf("Progress update interval: every %llu records\n", progress_interval);
    fflush(stdout);

    while (fread(&op, sizeof(PacketOut), 1, fp) == 1)
    {
        records_read++;
        if (records_read % progress_interval == 0ULL)
        {
            printf("Processed %llu packet records...\n", records_read);
            fflush(stdout);
        }
        etheta = op.etheta;
        pl_l1 = op.pl_l1;
        pl_l2 = op.pl_l2;
        pexit = op.pexit;
        phase = op.phase;
        nsteps = op.nscatt;

        reEf0 = op.Efe[0].re;
        imEf0 = op.Efe[0].im;
        reEf1 = op.Efe[1].re;
        imEf1 = op.Efe[1].im;

        reEr0 = op.Ere[0].re;
        imEr0 = op.Ere[0].im;
        reEr1 = op.Ere[1].re;
        imEr1 = op.Ere[1].im;

        for (j = 0; j < 3; j++)
        {
            mef[j] = op.mef[j];
            nef[j] = op.nef[j];
            sef[j] = op.sef[j];
            mer[j] = op.mer[j];
            ner[j] = op.ner[j];
            ser[j] = op.ser[j];
        }

        if (nsteps > 1)
        {
            weight = pexit * exp(-(ACOEFF_l1) * pl_l1) * exp(-(ACOEFF_l2) * pl_l2);
        }
        else
        {
            weight = 0.5 * pexit * exp(-(ACOEFF_l1) * pl_l1);
        }

        if (weight < SMALL)
            continue;

        ibin = (int)floor(etheta / dtheta);
        if (ibin < 0 || ibin >= NBINS)
            continue;

        /* Preserve the original reducer's phase folding exactly. */
        phase = acos(cos(phase));

        Efwd[0] = reEf0 + I * imEf0;
        Efwd[1] = reEf1 + I * imEf1;

        Erev[0] = reEr0 + I * imEr0;
        Erev[1] = reEr1 + I * imEr1;

        Sfe[0] = cabs(Efwd[0]) * cabs(Efwd[0]) + cabs(Efwd[1]) * cabs(Efwd[1]);
        Sfe[1] = cabs(Efwd[0]) * cabs(Efwd[0]) - cabs(Efwd[1]) * cabs(Efwd[1]);
        Sfe[2] = 2.0 * creal(Efwd[0] * conj(Efwd[1]));
        Sfe[3] = 2.0 * cimag(Efwd[0] * conj(Efwd[1]));

        Sre[0] = cabs(Erev[0]) * cabs(Erev[0]) + cabs(Erev[1]) * cabs(Erev[1]);
        Sre[1] = cabs(Erev[0]) * cabs(Erev[0]) - cabs(Erev[1]) * cabs(Erev[1]);
        Sre[2] = 2.0 * creal(Erev[0] * conj(Erev[1]));
        Sre[3] = 2.0 * cimag(Erev[0] * conj(Erev[1]));

        /* Rotate to common reference frame, see Ramella-Roman et al. */
        ep = atan2(nef[2], mef[2]);
        for (j = 0; j < 3; j++)
        {
            mrotf[j] = mef[j] * cos(ep) + nef[j] * sin(ep);
            nrotf[j] = -mef[j] * sin(ep) + nef[j] * cos(ep);
            srotf[j] = sef[j];
        }

        phi = atan2(nrotf[0], nrotf[1]);
        mrotf2[0] = cos(phi) * mrotf[0] - sin(phi) * mrotf[1];
        mrotf2[1] = sin(phi) * mrotf[0] + cos(phi) * mrotf[1];
        mrotf2[2] = mrotf[2];
        nrotf2[0] = cos(phi) * nrotf[0] - sin(phi) * nrotf[1];
        nrotf2[1] = sin(phi) * nrotf[0] + cos(phi) * nrotf[1];
        nrotf2[2] = nrotf[2];
        srotf2[0] = cos(phi) * srotf[0] - sin(phi) * srotf[1];
        srotf2[1] = sin(phi) * srotf[0] + cos(phi) * srotf[1];
        srotf2[2] = srotf[2];

        for (j = 0; j < 16; j++)
        {
            Rphi[j] = 0.0;
            Rep[j] = 0.0;
        }

        Rphi[0] = 1.0;
        Rphi[5] = cos(2.0 * phi);
        Rphi[6] = -sin(2.0 * phi);
        Rphi[9] = sin(2.0 * phi);
        Rphi[10] = cos(2.0 * phi);
        Rphi[15] = 1.0;

        Rep[0] = 1.0;
        Rep[5] = cos(2.0 * ep);
        Rep[6] = sin(2.0 * ep);
        Rep[9] = -sin(2.0 * ep);
        Rep[10] = cos(2.0 * ep);
        Rep[15] = 1.0;

        matrix_multiply(Rphi, Rep, Mrot);

        for (j = 0; j < 4; j++)
        {
            Sfrot[j] = 0.0;
            for (k = 0; k < 4; k++)
            {
                Sfrot[j] += Mrot[j * 4 + k] * Sfe[k];
            }
        }

        ep = atan2(ner[2], mer[2]);
        for (j = 0; j < 3; j++)
        {
            mrotr[j] = mer[j] * cos(ep) + ner[j] * sin(ep);
            nrotr[j] = -mer[j] * sin(ep) + ner[j] * cos(ep);
            srotr[j] = ser[j];
        }

        phi = atan2(nrotr[0], nrotr[1]);
        mrotr2[0] = cos(phi) * mrotr[0] - sin(phi) * mrotr[1];
        mrotr2[1] = sin(phi) * mrotr[0] + cos(phi) * mrotr[1];
        mrotr2[2] = mrotr[2];
        nrotr2[0] = cos(phi) * nrotr[0] - sin(phi) * nrotr[1];
        nrotr2[1] = sin(phi) * nrotr[0] + cos(phi) * nrotr[1];
        nrotr2[2] = nrotr[2];
        srotr2[0] = cos(phi) * srotr[0] - sin(phi) * srotr[1];
        srotr2[1] = sin(phi) * srotr[0] + cos(phi) * srotr[1];
        srotr2[2] = srotr[2];

        for (j = 0; j < 16; j++)
        {
            Rphi[j] = 0.0;
            Rep[j] = 0.0;
        }

        Rphi[0] = 1.0;
        Rphi[5] = cos(2.0 * phi);
        Rphi[6] = -sin(2.0 * phi);
        Rphi[9] = sin(2.0 * phi);
        Rphi[10] = cos(2.0 * phi);
        Rphi[15] = 1.0;

        Rep[0] = 1.0;
        Rep[5] = cos(2.0 * ep);
        Rep[6] = sin(2.0 * ep);
        Rep[9] = -sin(2.0 * ep);
        Rep[10] = cos(2.0 * ep);
        Rep[15] = 1.0;

        matrix_multiply(Rphi, Rep, Mrot);

        for (j = 0; j < 4; j++)
        {
            Srrot[j] = 0.0;
            for (k = 0; k < 4; k++)
            {
                Srrot[j] += Mrot[j * 4 + k] * Sre[k];
            }
        }

        /* Preserve the original field-space projection path, even though the
         * Stokes-rotation results Sfrot/Srrot are also computed. */
        Efrot[0] = Efwd[0] * dot3_arr(mef, mrotf2) + Efwd[1] * dot3_arr(nef, mrotf2);
        Efrot[1] = Efwd[0] * dot3_arr(mef, nrotf2) + Efwd[1] * dot3_arr(nef, nrotf2);

        Errot[0] = Erev[0] * dot3_arr(mer, mrotr2) + Erev[1] * dot3_arr(ner, mrotr2);
        Errot[1] = Erev[0] * dot3_arr(mer, nrotr2) + Erev[1] * dot3_arr(ner, nrotr2);

        If = cabs(Efrot[0]) * cabs(Efrot[0]) + cabs(Efrot[1]) * cabs(Efrot[1]);
        Qf = cabs(Efrot[0]) * cabs(Efrot[0]) - cabs(Efrot[1]) * cabs(Efrot[1]);
        Uf = 2.0 * creal(Efrot[0] * conj(Efrot[1]));
        Vf = 2.0 * cimag(Efrot[0] * conj(Efrot[1]));

        Ir = cabs(Errot[0]) * cabs(Errot[0]) + cabs(Errot[1]) * cabs(Errot[1]);
        Qr = cabs(Errot[0]) * cabs(Errot[0]) - cabs(Errot[1]) * cabs(Errot[1]);
        Ur = 2.0 * creal(Errot[0] * conj(Errot[1]));
        Vr = 2.0 * cimag(Errot[0] * conj(Errot[1]));

        incI[ibin] += weight * (If + Ir);
        incQ[ibin] += weight * (Qf + Qr);
        incU[ibin] += weight * (Uf + Ur);
        incV[ibin] += weight * (Vf + Vr);

        if (nsteps > 1)
        {
            polf = 1.0;
            polr = 1.0;

            E0 = (cos(phase) + I * sin(phase)) * Efrot[0] + Errot[0];
            E1 = (cos(phase) + I * sin(phase)) * Efrot[1] + Errot[1];

            SI[ibin] += weight * ((1.0 - polf) * If + (1.0 - polr) * Ir + cabs(E0) * cabs(E0) + cabs(E1) * cabs(E1));
            SQ[ibin] += weight * (cabs(E0) * cabs(E0) - cabs(E1) * cabs(E1));
            SU[ibin] += weight * 2.0 * creal(E0 * conj(E1));
            SV[ibin] += weight * 2.0 * cimag(E0 * conj(E1));

            count_ms[ibin] += weight;
        }
        else
        {
            SI[ibin] += weight * (If + Ir);
            SQ[ibin] += weight * (Qf + Qr);
            SU[ibin] += weight * (Uf + Ur);
            SV[ibin] += weight * (Vf + Vr);

            count_ss[ibin] += weight;
        }
    }

    if (!feof(fp))
    {
        fprintf(stderr, "Error while reading input file: %s\n", input_path);
        fclose(fp);
        return EXIT_FAILURE;
    }

    fclose(fp);

    printf("Finished reading %llu packet records. Writing reduced output...\n", records_read);
    fflush(stdout);

    fp = fopen(output_path, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open output file: %s\n", output_path);
        return EXIT_FAILURE;
    }

    for (i = 0; i < NBINS; i++)
    {
        ncount = count_ms[i] + count_ss[i];
        if (ncount <= 0.0)
            continue;

        fprintf(fp,
                "%e %e %e %e %e %e %e %e %e %e %e\n",
                (i + 0.5) * dtheta,
                SI[i] / ncount,
                SQ[i] / ncount,
                SU[i] / ncount,
                SV[i] / ncount,
                incI[i] / ncount,
                incQ[i] / ncount,
                incU[i] / ncount,
                incV[i] / ncount,
                count_ms[i],
                count_ss[i]);
    }

    fclose(fp);
    return 0;
}


