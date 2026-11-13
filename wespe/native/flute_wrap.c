/* Thin wrappers around CUGR2's Flute (LUT path).
 *
 * init_param / flutes_HD live in flute_mst.c and pull in the high-degree MST
 * package. Windows we repair have a handful of pins (typically 2), well inside
 * the POWV9/POST9 LUT (d <= 9) and flutes_MD range. We therefore stub those
 * two symbols: init_param is a no-op, and flutes_HD aborts if ever called.
 * This is still a real flute() + readLUT call, not the flute_style approximation.
 */
#include "flute.h"

void readLUT(void);
Tree flute(int d, DTYPE x[], DTYPE y[], int acc);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void init_param(void) {}

Tree flutes_HD(int d, DTYPE xs[], DTYPE ys[], int s[], int acc)
{
    (void)xs;
    (void)ys;
    (void)s;
    (void)acc;
    fprintf(stderr, "flute_wrap: flutes_HD is not linked (d=%d); use d small enough for LUT/MD\n", d);
    exit(2);
}

static int lut_ready = 0;

int flute_init(const char *lut_dir)
{
    char cwd[4096];
    if (lut_dir == NULL || lut_dir[0] == '\0')
        return -1;
    if (getcwd(cwd, sizeof cwd) == NULL)
        return -1;
    if (chdir(lut_dir) != 0)
        return -1;
    readLUT();
    if (chdir(cwd) != 0)
        return -1;
    lut_ready = 1;
    return 0;
}

int flute_rsmt(int d, const int *xs, const int *ys, int *out_len, int *out_nbranch,
               int *out_x, int *out_y, int *out_nbr)
{
    DTYPE x[MAXD], y[MAXD];
    int i, nb;
    Tree t;

    if (!lut_ready || d < 2 || d > MAXD)
        return -1;
    for (i = 0; i < d; i++) {
        x[i] = xs[i];
        y[i] = ys[i];
    }
    t = flute(d, x, y, ACCURACY);
    nb = t.deg + t.deg - 2;
    if (out_len)
        *out_len = t.length;
    if (out_nbranch)
        *out_nbranch = nb;
    for (i = 0; i < nb; i++) {
        out_x[i] = t.branch[i].x;
        out_y[i] = t.branch[i].y;
        out_nbr[i] = t.branch[i].n;
    }
    free(t.branch);
    return 0;
}
