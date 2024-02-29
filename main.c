#include <stdio.h>
#include "nn.h"


int main() {
    
    Matrix mat = mat_alloc(3, 3);
    mat_rand(mat, -1, 1);
    mat_print(mat);
        
    return 0;
}

