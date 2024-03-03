#include <stdio.h>
#include <time.h>
#include "src/blas/blas.h"
#include "src/layer.h"


int main() {
    
    srand(time(NULL));
    
    Matrix inputs = mat_alloc(10, 16);
    mat_rand(inputs, -10, 10);

    Layer layer1 = layer_alloc(16, 128, LEAKY_RELU);
    Layer layer2 = layer_alloc(128, 128, LEAKY_RELU);
    Layer layer3 = layer_alloc(128, 4, TANH);

    // Prediction
    printf("Inputs: \n");
    mat_print(inputs);
    printf("-------------------------------------------\n");
    printf("Outputs: \n");
    
    layer_forward(&layer1, inputs);
    layer_forward(&layer2, layer1.outputs);
    layer_forward(&layer3, layer2.outputs);

    mat_print(layer3.outputs);

    // Prediction 2
    printf("\n=============== PREDICTION 2 ==============\n");
    Matrix input_slice = mat_row_slice(inputs, 0, 5);

    printf("Inputs: \n");
    mat_print(input_slice);
    printf("-------------------------------------------\n");
    printf("Outputs: \n");
    
    layer_forward(&layer1, input_slice);
    layer_forward(&layer2, layer1.outputs);
    layer_forward(&layer3, layer2.outputs);

    mat_print(layer3.outputs);

    return 0;
}

