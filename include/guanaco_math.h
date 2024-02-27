#ifndef GUANACO_MATH_H_
#define GUANACO_MATH_H_

#define GUANACO_ARRAY_LEN(a) (sizeof((a))/sizeof((a)[0]))
#define GUANACO_MAT_AT(m, i, j) ((m).es[(i) * (m).stride + (j)])

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Matrix;

typedef enum {
  GUANACO_AF_SIGMOID,
  GUANACO_AF_RELU
} AF;

float guanaco_rand(void);
int guanaco_round(float x);
float guanaco_sigmoid(float x);
float guanaco_dx_sigmoid(float x);
float guanaco_relu(float x);
float guanaco_dx_relu(float x);
float guanaco_leaky_relu(float x);
float guanaco_dx_leaky_relu(float x);
Matrix guanaco_mat_create(size_t rows, size_t cols);
void guanaco_mat_fill(Matrix m, float x);
void guanaco_mat_rand(Matrix m, float low, float high);
Matrix guanaco_mat_row(Matrix m, size_t row);
void guanaco_mat_copy(Matrix dst, Matrix src);
void guanaco_mat_activate(Matrix m, AF af_id);
void guanaco_mat_add(Matrix dst, Matrix a);
void guanaco_mat_mult(Matrix dst, Matrix a, Matrix b);

#endif  // GUANACO_MATH_H_

#ifdef GUANACO_IMPLEMENTATION
#include "../src/guanaco_math.c"
#endif  // GUANACO_IMPLEMENTATION
