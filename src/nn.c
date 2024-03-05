#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "blas/blas.h"
#include "layer.h"
#include "nn.h"

NN nn_alloc(size_t n_layers, Layer *layers) {
    
    NN nn;
    
    nn.n_layers = n_layers;
    nn.layers = layers;

    // Outputs are assigned after each forward pass

    return nn;
}

void nn_forward(NN *nn, Matrix X) {

    Matrix outputs = X;

    for(size_t i = 0; i < nn->n_layers; i++) {
        layer_forward(&nn->layers[i], outputs);
        outputs = nn->layers[i].outputs;
    }

    nn->outputs = outputs;
}

void nn_backward(NN *nn, Matrix y, float learning_rate) {
    
    // TODO: Implement loss outside of here
    Matrix out_grad = mat_alloc(nn->outputs.rows, nn->outputs.cols);
    mat_copy(out_grad, nn->outputs);
    mat_sub(out_grad, y);
    scalar_mul(out_grad, 2);

     for(size_t i = nn->n_layers - 1; i > 0; i--) {
        layer_backward(&nn->layers[i], out_grad, learning_rate);
        out_grad = nn->layers[i].d_inputs;
    }
}

void nn_fit(NN *nn, Matrix X, Matrix y, size_t epochs, float learning_rate) {
    
    // TODO: Implement batching

    for(size_t i = 0; i < epochs; i++) {

        nn_forward(nn, X);
        nn_backward(nn, y, learning_rate);

        // TODO: Implement loss externally

    }
}

void nn_print(NN nn) {
    
    printf("NN[\n");
    
    for(size_t i = 0; i < nn.n_layers; i++) {
        printf("    ");
        layer_print(nn.layers[i]);
    }

    printf("]\n");
}

