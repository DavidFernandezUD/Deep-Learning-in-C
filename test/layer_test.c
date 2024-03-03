#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "../src/layer.h"

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

int main() {
    
    srand(time(NULL));
    
    test_layer_forward();

    printf("Layer test passed successfully!\n");

    return 0;
}

