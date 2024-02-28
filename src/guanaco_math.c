#ifndef GUANACO_IMPLEMENTATION
#include <guanaco_defines.h>
#include <guanaco_math.h>
#endif  // GUANACO_IMPLEMENTATION

GUANACO_API float guanaco_rand(void) {
  return (float) GUANACO_RAND() / (float) GUANACO_RAND_MAX;
}

GUANACO_API int guanaco_round(float x) {
  return x >= 0 ? (int) (x + 0.5f) : (int) (x - 0.5f);
}

GUANACO_API float guanaco_sigmoid(float x) {
  return 1 / (1 + GUANACO_EXP(-x));
}

GUANACO_API float guanaco_dx_sigmoid(float x) {
  return x * (1 - x);
}

GUANACO_API float guanaco_relu(float x) {
  return x > 0 ? x : 0.01f * x;
}

GUANACO_API float guanaco_dx_relu(float x) {
  return x >= 0 ? 1 : 0.01f;
}

GUANACO_API Matrix guanaco_mat_create(size_t rows, size_t cols) {
  Matrix m = {
    .rows = rows,
    .cols = cols,
    .stride = cols
  };
  m.es = GUANACO_MALLOC(sizeof(*m.es) * rows * cols);
  GUANACO_ASSERT(m.es);
  return m;
}

GUANACO_API void guanaco_mat_fill(Matrix m, float x) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      GUANACO_MAT_AT(m, i, j) = x;
    }
  }
}

GUANACO_API void guanaco_mat_rand(Matrix m, float low, float high) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      GUANACO_MAT_AT(m, i, j) = guanaco_rand() * (high - low) + low;
    }
  }
}

GUANACO_API Matrix guanaco_mat_row(Matrix m, size_t row) {
  return (Matrix) {
    .rows = 1,
    .cols = m.cols,
    .stride = m.stride,
    .es = &GUANACO_MAT_AT(m, row, 0)
  };
}

GUANACO_API void guanaco_mat_copy(Matrix dst, Matrix src) {
  GUANACO_ASSERT(dst.rows == src.rows);
  GUANACO_ASSERT(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      GUANACO_MAT_AT(dst, i, j) = GUANACO_MAT_AT(src, i, j);
    }
  }
}

GUANACO_API void guanaco_mat_activate(Matrix m, AF af_id) {
  float (*af)(float) = 0;
  switch (af_id) {
  case GUANACO_AF_SIGMOID:
    af = guanaco_sigmoid;
    break;
  case GUANACO_AF_RELU:
    af = guanaco_relu;
    break;
  }
  GUANACO_ASSERT(af);

  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      GUANACO_MAT_AT(m, i, j) = af(GUANACO_MAT_AT(m, i, j));
    }
  }
}

GUANACO_API void guanaco_mat_add(Matrix dst, Matrix a) {
  GUANACO_ASSERT(dst.rows == a.rows);
  GUANACO_ASSERT(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      GUANACO_MAT_AT(dst, i, j) += GUANACO_MAT_AT(a, i, j);
    }
  }
}

GUANACO_API void guanaco_mat_mult(Matrix dst, Matrix a, Matrix b) {
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
