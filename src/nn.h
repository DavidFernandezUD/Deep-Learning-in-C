#include <stdlib.h>
#include "blas/blas.h"
#include "layer.h"


#ifndef _NN_H
#define _NN_H


typedef struct {
    size_t n_layers;
    Layer *layers;
    Matrix outputs;
} NN;


// NN allocation and deallocation
NN nn_alloc(size_t n_layers, Layer *layers);


// NN operations
void nn_forward(NN *nn, Matrix X);

void nn_backward(NN *nn, Matrix y, float learning_rate);

void nn_fit(NN *nn, Matrix X, Matrix y, size_t epochs, float learning_rate);


// Utils
void nn_print(NN nn);

#endif // _NN_H

