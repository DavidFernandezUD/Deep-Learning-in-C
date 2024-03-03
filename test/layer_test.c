#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "../src/layer.h"


#define TOLERANCE 1e-5


void test_layer_forward() {
    
    Layer layer = layer_alloc(2, 3, IDENTITY);
    mat_fill(layer.weights, 1);
    mat_fill(layer.bias, 1);

    float data[] = {
        0, 0,
        1, 0,
        0, 1,
        1, 1
    };

    Matrix inputs = mat(4, 2, data);
    
    layer_forward(&layer, inputs);

    float expected[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {2, 2, 2},
        {3, 3, 3}
    };

    layer_forward(&layer, inputs);

    for(size_t i = 0; i < layer.outputs.rows; i++) {
        for(size_t j = 0; j < layer.outputs.cols; j++) {
            assert(MAT_AT(layer.outputs, i, j) == expected[i][j]);
        }
    }
    
    layer_free(layer);
}

void test_layer_backward() {
    
    Layer layer = layer_alloc(2, 3, IDENTITY);
    mat_fill(layer.weights, 0.5);

    float data[] = {
        0, 0,
        1, 0,
        0, 1,
        1, 1
    };
    Matrix inputs = mat(4, 2, data);

    layer_forward(&layer, inputs);

    float out_grad_data[] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    Matrix out_grad = mat(4, 3, out_grad_data);

    float expected_dw[][3] = {
        {0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5}
    };

    float expected_db[][3] = {
        {1, 1, 1}
    };

    float expected_dx[][2] = {
        {1.5, 1.5},
        {1.5, 1.5},
        {1.5, 1.5},
        {1.5, 1.5}
    };

    float expected_w[][3] = {
        {0, 0, 0},
        {0, 0, 0}
    };

    float expected_b[][3] = {
        {-1, -1, -1}
    };

    layer_backward(&layer, out_grad, 1);
    
    for(size_t i = 0; i < layer.d_weights.rows; i++) {
        for(size_t j = 0; j < layer.d_weights.cols; j++) {
            assert(MAT_AT(layer.d_weights, i, j) >= expected_dw[i][j] - TOLERANCE);
            assert(MAT_AT(layer.d_weights, i, j) <= expected_dw[i][j] + TOLERANCE);
        }
    }

    for(size_t i = 0; i < layer.d_bias.rows; i++) {
        for(size_t j = 0; j < layer.d_bias.cols; j++) {
            assert(MAT_AT(layer.d_bias, i, j) >= expected_db[i][j] - TOLERANCE);
            assert(MAT_AT(layer.d_bias, i, j) <= expected_db[i][j] + TOLERANCE);
        }
    }

    for(size_t i = 0; i < layer.d_inputs.rows; i++) {
        for(size_t j = 0; j < layer.d_inputs.cols; j++) {
            assert(MAT_AT(layer.d_inputs, i, j) >= expected_dx[i][j] - TOLERANCE);
            assert(MAT_AT(layer.d_inputs, i, j) <= expected_dx[i][j] + TOLERANCE);
        }
    }

     for(size_t i = 0; i < layer.weights.rows; i++) {
        for(size_t j = 0; j < layer.weights.cols; j++) {
            assert(MAT_AT(layer.weights, i, j) >= expected_w[i][j] - TOLERANCE);
            assert(MAT_AT(layer.weights, i, j) <= expected_w[i][j] + TOLERANCE);
        }
    }

     for(size_t i = 0; i < layer.bias.rows; i++) {
        for(size_t j = 0; j < layer.bias.cols; j++) {
            assert(MAT_AT(layer.bias, i, j) >= expected_b[i][j] - TOLERANCE);
            assert(MAT_AT(layer.bias, i, j) <= expected_b[i][j] + TOLERANCE);
        }
    }
}

int main() {
    
    srand(time(NULL));
    
    test_layer_forward();
    test_layer_backward();

    printf("Layer test passed successfully!\n");

    return 0;
}

