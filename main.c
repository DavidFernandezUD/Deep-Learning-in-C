#include <stdio.h>
#include "nn.h"


int main() {
    
    Matrix a = mat_alloc(3, 4);
    mat_fill(a, 2);

    Matrix b = mat_alloc(4, 5);
    mat_rand(b, -1, 1);

    Matrix c = mat_alloc(3, 5);

    mat_mul(c, a, b);
    mat_print(c);
        
    return 0;
}

