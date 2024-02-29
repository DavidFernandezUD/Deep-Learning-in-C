#ifndef _NN_H
#define _NN_H

#include <stdlib.h>


typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

#define MAT_AT(mat, i, j) (mat).data[(i) * (mat).cols + (j)]    // For accessing matrix indexes more easyly

// Utils
float rand_float();

float sigmoidf(float x);


// Matrix creation & initialization
Matrix mat_alloc(size_t rows, size_t cols);

void mat_fill(Matrix mat, float fill);

void mat_rand(Matrix mat, float low, float high);


// Matrix operations
void mat_mul(Matrix dest, Matrix a, Matrix b);

void mat_sum(Matrix dest, Matrix mat);

void mat_sig(Matrix mat);


// Matrix utils
void mat_print(Matrix mat);

#endif // _NN_H

