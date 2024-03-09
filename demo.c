#include <stdio.h>
#include <time.h>
#include "src/blas/blas.h"
#include "src/layer.h"
#include "src/nn.h"


int main() {

    srand(time(NULL));

    // <========= XOR MODEL =========>
    Layer layers[3] = {
        layer_alloc(2, 4, RELU),
        layer_alloc(4, 4, RELU),
        layer_alloc(4, 1, SIGMOID)
    };

    NN nn = nn_alloc(3, layers);

    nn_print(nn);

    float data[] = {
        0, 0,
        1, 0,
        0, 1,
        1, 1
    };
    Matrix X = mat(4, 2, data);

    float data2[] = {
        0,
        1,
        1,
        0
    };
    Matrix y = mat(4, 1, data2);

    nn_forward(&nn, X);
    printf("===== Outputs =====\n\n");
    mat_print(nn.outputs);

    // <========= TRAINING =========> 
    printf("\n\n===== TRAINING =====\n\n");
    nn_fit(&nn, X, y, 50000, 0.01f);

    nn_forward(&nn, X);
    printf("===== Outputs =====\n\n");
    mat_print(nn.outputs);
    
    return 0;
}

