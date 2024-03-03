#include <stdlib.h> 
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include "blas.h"


float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));  // expf() does exponentiation with float
}

float tanhf(float x) {
    float exp = expf(x);
    float nexp = expf(-x);
    return ((exp - nexp) / (exp + nexp));
}

float reluf(float x) {
    return (x > 0) ? x : 0;
}

float leaky_reluf(float x) {
    return (x > 0) ? x : x * LEAK_PARAM;
}

float actf(float x, Activation act) {
    switch(act) {
        case SIGMOID:       return sigmoidf(x); 
        case TANH:          return tanhf(x);
        case RELU:          return reluf(x);
        case LEAKY_RELU:    return leaky_reluf(x);
    }
    printf("ERROR: Invalid activation function\n");
    assert(0);
    return 0.0f;
}

float dactf(float y, Activation act) {
    switch(act) {
        case SIGMOID:       return y * (1 + y);
        case TANH:          return 1 - y * y;
        case RELU:          return (y > 0) ? 1 : 0;
        case LEAKY_RELU:    return (y > 0) ? 1 : LEAK_PARAM;
    }
    printf("ERROR: Invalid activation function\n");
    assert(0);
    return 0.0f;
}

Matrix mat_alloc(size_t rows, size_t cols) {    

    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.strides[0] = cols;
    mat.strides[1] = 1;
    mat.data = malloc(sizeof(*mat.data) * rows * cols);   // *mat.data instead of float for mantainability
    
    assert(mat.data != NULL);

    return mat;
}


Matrix mat(size_t rows, size_t cols, float* data) {
    
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.strides[0] = cols;
    mat.strides[1] = 1;
    mat.data = data;

    assert(mat.data != NULL);

    return mat;   
}


void mat_fill(Matrix mat, float fill) {
     
    for(size_t i = 0; i <mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = fill;
        }
    }   
}

void mat_rand(Matrix mat, float low, float high) {
    
    for(size_t i = 0; i <mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_copy(Matrix dest, const Matrix src) {

    assert(dest.rows == src.rows);
    assert(dest.cols == src.cols);

    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
}

Matrix mat_row(const Matrix mat, size_t row) {
    return (Matrix) {
        .rows = 1,
        .cols = mat.cols,
        .strides[0] = mat.strides[0],
        .strides[1] = mat.strides[1],
        .data = &MAT_AT(mat, row, 0)  
    };
}

Matrix mat_row_slice(const Matrix mat, size_t start_row, size_t end_row) {
    
    assert(start_row >= 0 && start_row < mat.rows);
    assert(end_row > start_row && end_row <= mat.rows);

    return (Matrix) {
      .rows = end_row - start_row,
      .cols = mat.cols,
      .strides[0] = mat.strides[0],
      .strides[1] = mat.strides[1],
      .data = &MAT_AT(mat, start_row, 0)
    };
}

Matrix mat_col_slice(const Matrix mat, size_t start_col, size_t end_col) {
    
    assert(start_col >= 0 && start_col < mat.cols);
    assert(end_col > start_col && end_col <= mat.cols);

    return (Matrix) {
        .rows = mat.rows,
        .cols = end_col - start_col,
        .strides[0] = mat.strides[0],
        .strides[1] = mat.strides[1],
        .data = &MAT_AT(mat, 0, start_col)
    };
}

Matrix mat_slice(const Matrix mat, size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    
    assert(start_row >= 0 && start_row < mat.rows);
    assert(end_row > start_row && end_row <= mat.rows);

    assert(start_col >= 0 && start_col < mat.cols);
    assert(end_col > start_col && end_col <= mat.cols);

    return (Matrix) {
        .rows = end_row - start_row,
        .cols = end_col - start_col,
        .strides[0] = mat.strides[0],
        .strides[1] = mat.strides[1],
        .data = &MAT_AT(mat, start_row, start_col)
    };
}

Matrix transpose(Matrix mat) {
    return (Matrix) {
        .rows = mat.cols,
        .cols = mat.rows,
        .strides[0] = mat.strides[1],
        .strides[1] = mat.strides[0],
        .data = mat.data
    };
}

void mat_mul(Matrix dest, const Matrix a, const Matrix b) {
    
    assert(a.cols == b.rows);
    assert(dest.rows == a.rows);
    assert(dest.cols == b.cols);
    
    mat_fill(dest, 0);

    for(size_t i = 0; i < a.rows; i++) {
        for(size_t k = 0; k < a.cols; k++) {
            for(size_t j = 0; j < b.cols; j++) {
                MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
           }
        }
    }
}

void hadamard(Matrix dest, const Matrix mat) {
  
    bool isVector = (mat.rows == 1);

    assert(dest.rows == mat.rows || isVector);
    assert(dest.cols == mat.cols);

    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) *= (isVector) ? MAT_AT(mat, 0, j) : MAT_AT(mat, i, j);
        }
    }
}

void mat_sum(Matrix dest, const Matrix mat) {
    
    bool isVector = (mat.rows == 1);

    assert(dest.rows == mat.rows || isVector);
    assert(dest.cols == mat.cols);

    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) += (isVector) ? MAT_AT(mat, 0, j) : MAT_AT(mat, i, j);
        }
    }
}

void mat_sub(Matrix dest, const Matrix mat) {
    
    bool isVector = (mat.rows == 1);

    assert(dest.rows == mat.rows || isVector);
    assert(dest.cols == mat.cols);

    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) -= (isVector) ? MAT_AT(mat, 0, j) : MAT_AT(mat, i, j);
        }
    }
}

void scalar_add(Matrix mat, float scalar) {
    
    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) += scalar;
        }
    }
}

void scalar_mul(Matrix mat, float scalar) {
     
    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) *= scalar;
        }
    }   
}

void mat_act(Matrix mat, Activation act) {
    
    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = actf(MAT_AT(mat, i, j), act);
        }
    }
}

void mat_dact(Matrix mat, Activation act) {

    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = dactf(MAT_AT(mat, i, j), act);
        }
    }
}

void mat_mean(Matrix dest, const Matrix mat, int axis) {
    
    float n;
    if(axis == 0) {
        assert(dest.cols == mat.cols);
        assert(dest.rows == 1);
        n = mat.rows;
    } else if(axis == 1) {
        assert(dest.rows == mat.rows);
        assert(dest.cols == 1);
        n = mat.cols;
    } else {
        printf("ERROR: Invalid axis\n");
        assert(0);
    }

    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            if(axis == 0) {
                MAT_AT(dest, 0, j) += MAT_AT(mat, i, j) / n;
            } else {
                MAT_AT(dest, i, 0) += MAT_AT(mat, i, j) / n;
            }
        } 
    }
}

void mat_print(Matrix mat) {

    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            printf("%9.6f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
}

