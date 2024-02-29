#include "nn.h"
#include <stdlib.h> 
#include <stdio.h>
#include <assert.h>


Matrix mat_alloc(size_t rows, size_t cols) {    
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = malloc(sizeof(*mat.data) * rows * cols);   // *mat.data instead of float for mantainability
    
    assert(mat.data != NULL);
    return mat;
}


float rand_float() {
    return (float) rand() / (float) RAND_MAX;
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

void mat_mul(Matrix dest, Matrix a, Matrix b) {
    
    assert(a.cols == b.rows);
    assert(dest.rows == a.rows);
    assert(dest.cols == b.cols);

    for(size_t i = 0; i < a.rows; i++) {
        for(size_t j = 0; j < b.cols; j++) {
            MAT_AT(dest, i, j) = 0;
            for(size_t k = 0; k < a.cols; k++) {
                MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Matrix dest, Matrix mat) {
    
    assert(dest.rows == mat.rows);
    assert(dest.cols == mat.cols);

    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(dest, i, j) += MAT_AT(mat, i, j);
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

