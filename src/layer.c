#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "blas/blas.h"
#include "layer.h"


Layer layer_alloc(size_t n_inputs, size_t n_outputs, Activation act) {
    
    Layer layer;

    // Layer shape
    layer.n_inputs = n_inputs;
    layer.n_outputs = n_outputs;

    // Allocatiom and initialization of parameters
    layer.weights = mat_alloc(n_inputs, n_outputs);
    layer.bias = mat_alloc(1, n_outputs);
    
    // Allocation of gradients
    layer.d_weights = mat_alloc(n_inputs, n_outputs);
    layer.d_bias = mat_alloc(1, n_outputs);
    
    // TODO: Initialize weights based on previous layer n_outputs
    mat_rand(layer.weights, -1, 1);
    mat_fill(layer.bias, 0);
    
    // Layer activation
    switch(act) {
        case SIGMOID:
            layer.act = act;
            break;
        case TANH:
            layer.act = act;
            break;
        case RELU:
            layer.act = act;
            break;
        case LEAKY_RELU:
            layer.act = act;
            break;
        case IDENTITY:
            layer.act = act;
            break;
        default:
            layer.act = IDENTITY;
            fprintf(stderr, "[ERROR] Invalid activation. Activation defaulted to IDENTITY\n");
    }
    
    // Outputs and d_inputs matrices may be reallocated in forward and backward pass
    layer.outputs = mat_alloc(1, n_outputs);
    layer.d_inputs = mat_alloc(1, n_inputs);

    return layer;
}

void layer_free(Layer layer) {
    mat_free(layer.weights);
    mat_free(layer.bias);
    mat_free(layer.d_weights);
    mat_free(layer.d_bias);
    mat_free(layer.d_inputs);
    mat_free(layer.outputs); 
}

void layer_forward(Layer *layer, const Matrix inputs) {
    
    assert(inputs.cols == layer->n_inputs);

    layer->inputs = inputs;
    
    // Outputs matrix is realocated if batch size changes
    if(layer->outputs.rows != inputs.rows) {
        mat_free(layer->outputs);
        layer->outputs = mat_alloc(inputs.rows, layer->n_outputs);
    }

    mat_mul(layer->outputs, inputs, layer->weights);
    mat_sum(layer->outputs, layer->bias);
    mat_act(layer->outputs, layer->act);
}

void layer_backward(Layer *layer, Matrix out_grad, float learning_rate) {
    
    assert(layer->outputs.cols == out_grad.cols);

    // d_inputs matrix is realocated if batch size changes
    if(layer->d_inputs.rows != out_grad.rows) {
        mat_free(layer->d_inputs);
        layer->d_inputs = mat_alloc(out_grad.rows, layer->n_inputs);
    }

    // dL/dz
    mat_dact(layer->outputs, layer->act);
    hadamard(out_grad, layer->outputs);

    // dL/dw
    Matrix inputs_t = transpose(layer->inputs);
    mat_mul(layer->d_weights, inputs_t, out_grad);
    scalar_mul(layer->d_weights, (1.0f / out_grad.rows));

    // dL/db
    mat_mean(layer->d_bias, out_grad, 0);

    // dL/dX
    Matrix weights_t = transpose(layer->weights);
    mat_mul(layer->d_inputs, out_grad, weights_t);

    // Update parameters
    scalar_mul(layer->d_weights, learning_rate);
    scalar_mul(layer->d_bias, learning_rate);

    mat_sub(layer->weights, layer->d_weights);
    mat_sub(layer->bias, layer->d_bias);

}

void layer_print(Layer layer) {
    
    char act_name[20];
    switch(layer.act) {
        case SIGMOID:
            strcpy(act_name, "sigmoid");
            break;
        case TANH:
            strcpy(act_name, "tanh");
            break;
        case RELU:
            strcpy(act_name, "ReLU");
            break;
        case LEAKY_RELU:
            strcpy(act_name, "Leaky ReLU");
            break;
        case IDENTITY:
            strcpy(act_name, "Identity");
            break;
    }

    printf("Layer[n_inputs: %ld, n_outputs: %ld, activation: %s]\n",
            layer.n_inputs, layer.n_outputs, act_name
    );
}

