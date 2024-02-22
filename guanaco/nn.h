#ifndef GUANACO_NN_H_
#define GUANACO_NN_H_

#define GUANACO_NN_INPUT(net) ((net).as[0])
#define GUANACO_NN_OUTPUT(net) ((net).as[(net).count])

typedef struct {
  size_t count;
  AF af_id;
  Matrix *ws;
  Matrix *bs;
  Matrix *as;
} NN;

NN guanaco_nn_create(size_t *arch, size_t arch_count, AF af_id);
void guanaco_nn_rand(NN net, float lower, float upper);
void guanaco_nn_zero(NN net);
void guanaco_nn_forward(NN net);
float guanaco_nn_cost(NN net, Matrix Xs, Matrix Ys);
void guanaco_nn_finite_diff(NN net,
                            NN grad,
                            float eps,
                            Matrix Xs,
                            Matrix Ys);
void guanaco_nn_backprop(NN net,
                         NN grad,
                         Matrix Xs,
                         Matrix Ys);
void guanaco_nn_learn(NN net, NN grad, float rate);

#endif  // GUANACO_NN_H_

#ifdef GUANACO_IMPLEMENTATION

inline NN guanaco_nn_create(size_t *arch, size_t arch_count, AF af_id) {
  GUANACO_ASSERT(arch_count);

  NN net = {
    .count = arch_count - 1,
    .af_id = af_id
  };
  net.ws = GUANACO_MALLOC(sizeof(*net.ws) * net.count);
  GUANACO_ASSERT(net.ws);
  net.bs = GUANACO_MALLOC(sizeof(*net.bs) * net.count);
  GUANACO_ASSERT(net.bs);
  net.as = GUANACO_MALLOC(sizeof(*net.as) * (net.count + 1));
  GUANACO_ASSERT(net.as);

  net.as[0] = guanaco_mat_create(1, arch[0]);
  for (size_t i = 1; i < arch_count; ++i) {
    net.ws[i - 1] = guanaco_mat_create(net.as[i - 1].cols, arch[i]);
    net.bs[i - 1] = guanaco_mat_create(1, arch[i]);
    net.as[i]     = guanaco_mat_create(1, arch[i]);
  }

  return net;
}

inline void guanaco_nn_rand(NN net, float lower, float upper) {
  for (size_t i = 0; i < net.count; ++i) {
    guanaco_mat_rand(net.ws[i], lower, upper);
    guanaco_mat_rand(net.bs[i], lower, upper);
  }
}

inline void guanaco_nn_zero(NN net) {
  for (size_t i = 0; i < net.count; ++i) {
    guanaco_mat_fill(net.ws[i], 0);
    guanaco_mat_fill(net.bs[i], 0);
    guanaco_mat_fill(net.as[i], 0);
  }
  guanaco_mat_fill(net.as[net.count], 0);
}

inline void guanaco_nn_forward(NN net) {
  for (size_t i = 0; i < net.count; ++i) {
    guanaco_mat_mult(net.as[i + 1], net.as[i], net.ws[i]);
    guanaco_mat_add(net.as[i + 1], net.bs[i]);
    guanaco_mat_activate(net.as[i + 1], net.af_id);
  }
}

inline float guanaco_nn_cost(NN net, Matrix Xs, Matrix Ys) {
  GUANACO_ASSERT(Xs.rows == Ys.rows);
  GUANACO_ASSERT(Ys.cols == GUANACO_NN_OUTPUT(net).cols);
  float c = 0;
  size_t n = Xs.rows;
  size_t q = Ys.cols;
  for (size_t i = 0; i < n; ++i) {
    Matrix x = guanaco_mat_row(Xs, i);
    Matrix y = guanaco_mat_row(Ys, i);
    guanaco_mat_copy(GUANACO_NN_INPUT(net), x);
    guanaco_nn_forward(net);
    for (size_t j = 0; j < q; ++j) {
      float d = GUANACO_MAT_AT(GUANACO_NN_OUTPUT(net), 0, j) - GUANACO_MAT_AT(y, 0, j);
      c += d * d;
    }
  }
  return c / n;
}

inline void guanaco_nn_finite_diff(NN net,
                                   NN grad,
                                   float eps,
                                   Matrix Xs,
                                   Matrix Ys) {
  float saved = 0;
  float c = guanaco_nn_cost(net, Xs, Ys);
  for (size_t i = 0; i < net.count; ++i) {
    for (size_t j = 0; j < net.ws[i].rows; ++j) {
      for (size_t k = 0; k < net.ws[i].cols; ++k) {
        saved = GUANACO_MAT_AT(net.ws[i], j, k);
        GUANACO_MAT_AT(net.ws[i], j, k) += eps;
        GUANACO_MAT_AT(grad.ws[i], j, k) = (guanaco_nn_cost(net, Xs, Ys) - c) / eps;
        GUANACO_MAT_AT(net.ws[i], j, k) = saved;
      }
    }
    for (size_t j = 0; j < net.bs[i].rows; ++j) {
      for (size_t k = 0; k < net.bs[i].cols; ++k) {
        saved = GUANACO_MAT_AT(net.bs[i], j, k);
        GUANACO_MAT_AT(net.bs[i], j, k) += eps;
        GUANACO_MAT_AT(grad.bs[i], j, k) = (guanaco_nn_cost(net, Xs, Ys) - c) / eps;
        GUANACO_MAT_AT(net.bs[i], j, k) = saved;
      }
    }
  }
}

inline void guanaco_nn_backprop(NN net,
                                NN grad,
                                Matrix Xs,
                                Matrix Ys) {
  GUANACO_ASSERT(Xs.rows == Ys.rows);
  size_t n = Xs.rows;
  GUANACO_ASSERT(GUANACO_NN_OUTPUT(net).cols == Ys.cols);
  guanaco_nn_zero(grad);

  for (size_t i = 0; i < n; ++i) {
    // Feed-forward
    guanaco_mat_copy(GUANACO_NN_INPUT(net), guanaco_mat_row(Xs, i));
    guanaco_nn_forward(net);
    for (size_t j = 0; j <= net.count; ++j) {
      guanaco_mat_fill(grad.as[j], 0);
    }
    for (size_t j = 0; j < Ys.cols; ++j) {
      GUANACO_MAT_AT(GUANACO_NN_OUTPUT(grad), 0, j) = GUANACO_MAT_AT(GUANACO_NN_OUTPUT(net), 0, j) - GUANACO_MAT_AT(Ys, i, j);
    }
    // Back-propagation
    for (size_t l = net.count; l > 0; --l) {
      for (size_t j = 0; j < net.as[l].cols; ++j) {
        float a  = GUANACO_MAT_AT(net.as[l], 0, j);
        float da = GUANACO_MAT_AT(grad.as[l], 0, j);
        float derivative = 2 * da * a * (1 - a);
        GUANACO_MAT_AT(grad.bs[l - 1], 0, j) += derivative;
        for (size_t k = 0; k < net.as[l - 1].cols; ++k) {
          float pa = GUANACO_MAT_AT(net.as[l - 1], 0, k);
          float w  = GUANACO_MAT_AT(net.ws[l - 1], k, j);
          GUANACO_MAT_AT(grad.ws[l - 1], k, j) += derivative * pa;
          GUANACO_MAT_AT(grad.as[l - 1], 0, k) += derivative * w;
        }
      }
    }
  }
  // Average
  for (size_t i = 0; i < grad.count; ++i) {
    for (size_t j = 0; j < grad.ws[i].rows; ++j) {
      for (size_t k = 0; k < grad.ws[i].cols; ++k) {
        GUANACO_MAT_AT(grad.ws[i], j, k) /= n;
      }
    }
    for (size_t j = 0; j < grad.bs[i].rows; ++j) {
      for (size_t k = 0; k < grad.bs[i].cols; ++k) {
        GUANACO_MAT_AT(grad.bs[i], j, k) /= n;
      }
    }
  }
}

inline void guanaco_nn_learn(NN net, NN grad, float rate) {
  for (size_t i = 0; i < net.count; ++i) {
    for (size_t j = 0; j < net.ws[i].rows; ++j) {
      for (size_t k = 0; k < net.ws[i].cols; ++k) {
        GUANACO_MAT_AT(net.ws[i], j, k) -= rate * GUANACO_MAT_AT(grad.ws[i], j, k);
      }
    }
    for (size_t j = 0; j < net.bs[i].rows; ++j) {
      for (size_t k = 0; k < net.bs[i].cols; ++k) {
        GUANACO_MAT_AT(net.bs[i], j, k) -= rate * GUANACO_MAT_AT(grad.bs[i], j, k);
      }
    }
  }
}

#endif  // GUANACO_IMPLEMENTATION
