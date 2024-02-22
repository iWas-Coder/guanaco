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
  GUANACO_AF_RELU,
  GUANACO_AF_LEAKY_RELU
} AF;

float guanaco_rand(void);
float guanaco_sigmoid(float x);
float guanaco_relu(float x);
float guanaco_leaky_relu(float x);
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

inline float guanaco_rand(void) {
  return (float) GUANACO_RAND() / (float) GUANACO_RAND_MAX;
}

inline float guanaco_sigmoid(float x) {
  return 1.0f / (1.0f + GUANACO_EXP(-x));
}

inline float guanaco_relu(float x) {
  return GUANACO_MAX(0.0f, x);
}

inline float guanaco_leaky_relu(float x) {
  return (x > 0) ? x : 0.01f * x;
}

inline Matrix guanaco_mat_create(size_t rows, size_t cols) {
  Matrix m = {
    .rows = rows,
    .cols = cols,
    .stride = cols
  };
  m.es = GUANACO_MALLOC(sizeof(*m.es) * rows * cols);
  GUANACO_ASSERT(m.es);
  return m;
}

inline void guanaco_mat_fill(Matrix m, float x) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      GUANACO_MAT_AT(m, i, j) = x;
    }
  }
}

inline void guanaco_mat_rand(Matrix m, float low, float high) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      GUANACO_MAT_AT(m, i, j) = guanaco_rand() * (high - low) + low;
    }
  }
}

inline Matrix guanaco_mat_row(Matrix m, size_t row) {
  return (Matrix) {
    .rows = 1,
    .cols = m.cols,
    .stride = m.stride,
    .es = &GUANACO_MAT_AT(m, row, 0)
  };
}

inline void guanaco_mat_copy(Matrix dst, Matrix src) {
  GUANACO_ASSERT(dst.rows == src.rows);
  GUANACO_ASSERT(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      GUANACO_MAT_AT(dst, i, j) = GUANACO_MAT_AT(src, i, j);
    }
  }
}

inline void guanaco_mat_activate(Matrix m, AF af_id) {
  float (*af)(float) = 0;
  switch (af_id) {
  case GUANACO_AF_SIGMOID:
    af = guanaco_sigmoid;
    break;
  case GUANACO_AF_RELU:
    af = guanaco_relu;
    break;
  case GUANACO_AF_LEAKY_RELU:
    af = guanaco_leaky_relu;
    break;
  }
  GUANACO_ASSERT(af);

  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      GUANACO_MAT_AT(m, i, j) = af(GUANACO_MAT_AT(m, i, j));
    }
  }
}

inline void guanaco_mat_add(Matrix dst, Matrix a) {
  GUANACO_ASSERT(dst.rows == a.rows);
  GUANACO_ASSERT(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      GUANACO_MAT_AT(dst, i, j) += GUANACO_MAT_AT(a, i, j);
    }
  }
}

inline void guanaco_mat_mult(Matrix dst, Matrix a, Matrix b) {
  GUANACO_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  GUANACO_ASSERT(dst.rows == a.rows);
  GUANACO_ASSERT(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      GUANACO_MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        GUANACO_MAT_AT(dst, i, j) += GUANACO_MAT_AT(a, i, k) * GUANACO_MAT_AT(b, k, j);
      }
    }
  }
}

#endif  // GUANACO_IMPLEMENTATION
