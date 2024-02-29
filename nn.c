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

void mat_rand(Matrix mat, float low, float high) {
    
    for(size_t i = 0; i <mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_dot(Matrix dest, Matrix a, Matrix b) {
    
}

void mat_sum(Matrix dest, Matrix mat) {

}

void mat_print(Matrix mat) {

    for(size_t i = 0; i < mat.rows; i++) {
        for(size_t j = 0; j < mat.cols; j++) {
            printf("%9.6f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
}

