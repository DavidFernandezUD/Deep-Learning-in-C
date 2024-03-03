#include <stdio.h>
#include <time.h>
#include "blas.h"


int main() {
    
    srand(time(NULL));

    // Model
    Matrix w1 = mat_alloc(2, 2);
    mat_rand(w1, -1, 1);
    Matrix b1 = mat_alloc(1, 2);
    mat_rand(b1, -1, 1);
    Matrix z1 = mat_alloc(1, 2);

    Matrix w2 = mat_alloc(2, 1);
    mat_rand(w2, -1, 1);
    Matrix b2 = mat_alloc(1, 1);
    mat_rand(b2, -1, 1);
    Matrix z2 = mat_alloc(1, 1);

    // Inputs
    Matrix inputs = mat_alloc(1, 2);
    mat_rand(inputs, -1, 1);

    // Prediction
    printf("Inputs: \n");
    mat_print(inputs);
    printf("--------------------\n");
    printf("Outputs: \n");
    
    mat_mul(z1, inputs, w1);
    mat_sum(z1, b1);
    mat_act(z1, SIGMOID);

    mat_mul(z2, z1, w2);
    mat_sum(z2, b2);
    mat_act(z2, TANH);

    mat_print(z2);

    return 0;
}

