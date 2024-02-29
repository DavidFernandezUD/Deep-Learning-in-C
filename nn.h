#ifndef _NN_H
#define _NN_H

#include <stdlib.h>


typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

#define MAT_AT(mat, i, j) (mat).data[(i) * (mat).cols + (j)]    // For accessing matrix indexes more easyly

float rand_float();

Matrix mat_alloc(size_t rows, size_t cols);

void mat_rand(Matrix mat, float low, float high);

void mat_dot(Matrix dest, Matrix a, Matrix b);

void mat_add(Matrix dest, Matrix mat);

void mat_print(Matrix mat);

#endif // _NN_H

