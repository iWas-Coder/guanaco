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
                         Matrix Ys,
                         bool_t traditional);
void guanaco_nn_learn(NN net, NN grad, float rate);
void guanaco_nn_fit(NN net,
                    size_t *net_arch,
                    Matrix Xs,
                    Matrix Ys,
                    size_t epochs,
                    bool_t verbose);

#endif  // GUANACO_NN_H_

#ifdef GUANACO_IMPLEMENTATION
#include "../src/guanaco_nn.c"
#endif  // GUANACO_IMPLEMENTATION
