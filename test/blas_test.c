#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "../src/blas/blas.h"


void test_mat() {
    
    float data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    Matrix m = mat(3, 3, data);
    
    assert(m.rows == 3 && m.cols == 3 && m.strides[0] == 3);

    assert(MAT_AT(m, 1, 0) == 4);
    assert(MAT_AT(m, 0, 1) == 2);
    assert(MAT_AT(m, 1, 1) == 5);
    assert(MAT_AT(m, 2, 1) == 8);
}

void test_slice() {

    float data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    Matrix m = mat(3, 3, data);
    
    Matrix slice;

    slice = mat_slice(m, 1, 2, 0, 3);
    assert(slice.rows == 1 && slice.cols == 3);
    for(size_t i = 0; i < slice.cols; i++) {
        assert(MAT_AT(slice, 0, i) == MAT_AT(m, 1, i));
    }

    slice = mat_slice(m, 1, 3, 1, 3);
    assert(slice.rows == 2 && slice.cols == 2);
    for(size_t i = 0; i < slice.rows; i++) {
        for(size_t j = 0; j < slice.cols; j++) {
            assert(MAT_AT(slice, i, j) == MAT_AT(m, i+1, j+1));
        }
    }
}

void test_mat_mul() {
 
    float data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float expected[][3] = {
       {30,  36,  42},
       {66,  81,  96},
       {102, 126, 150} 
    };

    Matrix m = mat(3, 3, data);
    
    Matrix dest = mat_alloc(3, 3);
    mat_fill(dest, 999);

    mat_mul(dest, m, m);
    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            assert(MAT_AT(dest, i, j) == expected[i][j]);
        }
    }
}

void test_hadamard() {
 
    float data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    Matrix m = mat(3, 3, data);
    Matrix slice = mat_row_slice(m, 1, 2);

    Matrix dest = mat_alloc(3, 3);
    
    mat_fill(dest, 2);
    hadamard(dest, m);
    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            assert(MAT_AT(dest, i, j) == MAT_AT(m, i, j) * 2);
        }
    }

    mat_fill(dest, 2);
    hadamard(dest, slice);
    for(size_t i = 0; i < dest.rows; i++) {
        for(size_t j = 0; j < dest.cols; j++) {
            assert(MAT_AT(dest, i, j) == MAT_AT(m, 1, j) * 2);
        }
    }
}

void test_mat_act() {
    
    float data[] = {
       1, -2, -3,
       4, -5,  0,
      -7,  8,  9
    };

    float expected[][3] = {
      {1, 0, 0},
      {4, 0, 0},
      {0, 8, 9}  
    };

    Matrix m = mat(3, 3, data);
    mat_act(m, RELU);

    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            assert(MAT_AT(m, i, j) == expected[i][j]);
        }
    }
}

void test_mat_dact() {
    
    float data[] = {
       1, 0, 0,
       4, 0, 0,
       0, 8, 9
    };

    float expected[][3] = {
      {1, 0, 0},
      {1, 0, 0},
      {0, 1, 1}  
    };

    Matrix m = mat(3, 3, data);
    mat_dact(m, RELU);

    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            assert(MAT_AT(m, i, j) == expected[i][j]);
        }
    }
}

void test_mat_mean() {
  
    float data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float row_expected[] = {2, 5, 8};
    float col_expected[] = {4, 5, 6};

    Matrix m = mat(3, 3, data);

    Matrix row_mean = mat_alloc(3, 1);
    Matrix col_mean = mat_alloc(1, 3);

    mat_mean(row_mean, m, 1);
    mat_mean(col_mean, m, 0);
    
    for(size_t i = 0; i < 3; i++) {
        assert(MAT_AT(row_mean, i, 0) == row_expected[i]);
        assert(MAT_AT(col_mean, 0, i) == col_expected[i]);
    }
}

void test_transpose() {
    
    float data[] = {
       1, 2,
       3, 4,
       5, 6
    };

    float expected[][3] = {
      {1, 3, 5},
      {2, 4, 6},  
    };

    Matrix m = mat(3, 2, data);

    Matrix t = transpose(m);
    assert(t.rows == 2 && t.cols == 3);

    for(size_t i = 0; i < t.rows; i++) {
        for(size_t j = 0; j < t.cols; j++) {
            assert(MAT_AT(t, i, j) == expected[i][j]);
        }
    }
}

int main() {
    
    srand(time(NULL));

    test_mat();
    test_slice();
    test_mat_mul();
    test_hadamard();
    test_mat_act();
    test_mat_dact();
    test_mat_mean();
    test_transpose();

    printf("Blas test passed successfully!\n");

    return 0;
}

