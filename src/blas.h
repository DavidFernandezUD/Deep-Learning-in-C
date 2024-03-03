#include <stdlib.h>


#ifndef _BLAS_H
#define _BLAS_H


#define MAT_AT(mat, i, j) (mat).data[(i) * (mat).strides[0] + (j) * (mat).strides[1]]    // For accessing matrix indexes more easyly
#define LEAK_PARAM 0.01f


typedef struct {
    size_t rows;
    size_t cols;
    size_t strides[2];
    float *data;
} Matrix;

typedef enum {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU
} Activation;


// Utils
float rand_float();

float sigmoidf(float x);

float tanhf(float x);

float reluf(float x);

float leaky_reluf(float x);

float actf(float x, Activation act);

float dactf(float x, Activation act);
    

// Matrix creation & initialization
Matrix mat_alloc(size_t rows, size_t cols);

void mat_free(Matrix mat);

Matrix mat(size_t rows, size_t cols, float* data);

void mat_fill(Matrix mat, float fill);

void mat_rand(Matrix mat, float low, float high);

void mat_copy(Matrix dest, const Matrix src);

Matrix mat_row(const Matrix mat, size_t row);

Matrix mat_row_slice(const Matrix mat, size_t start_row, size_t end_row);

Matrix mat_col_slice(const Matrix mat, size_t start_col, size_t end_col);

Matrix mat_slice(const Matrix mat, size_t start_row, size_t end_row, size_t start_col, size_t end_col);

Matrix transpose(Matrix mat);


// Matrix operations
void mat_mul(Matrix dest, const Matrix a, const Matrix b);

void hadamard(Matrix dest, const Matrix mat);

void mat_sum(Matrix dest, const Matrix mat);

void mat_sub(Matrix dest, const Matrix mat);

void scalar_add(Matrix mat, float scalar);

void scalar_mul(Matrix mat, float scalar);

void mat_act(Matrix mat, Activation act);

void mat_dact(Matrix mat, Activation act);

void mat_mean(Matrix dest, const Matrix mat, int axis);


// Matrix utils
void mat_print(Matrix mat);

#endif // _BLAS_H

