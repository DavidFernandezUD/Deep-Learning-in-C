#include <stdlib.h>
#include <stdbool.h>
#include "blas/blas.h"


#ifndef _LAYER_H
#define _LAYER_H


typedef struct {
    size_t n_inputs;
    size_t n_outputs;
    Matrix weights;
    Matrix bias;
    Activation act;
    Matrix outputs;         // Initialized when receiving batch
} Layer;


// Layer creation and initialization
Layer layer_alloc(size_t n_inputs, size_t n_outputs, Activation act);

void layer_free(Layer layer);


// Layer operations
void layer_forward(Layer *layer, const Matrix inputs);

void layer_backward(Layer *layer, const Matrix out_grad);


// Layer utils
void layer_print(Layer layer);

#endif // _LAYER_H

