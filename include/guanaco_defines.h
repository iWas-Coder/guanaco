#pragma once

#include <stddef.h>

#ifdef GUANACO_IMPLEMENTATION
#define GUANACO_API inline
#else
#define GUANACO_API
#endif  // GUANACO_IMPLEMENTATION

#ifndef GUANACO_MALLOC
#include <stdlib.h>
#define GUANACO_MALLOC malloc
#endif // NN_MALLOC

#ifndef GUANACO_RAND
#include <stdlib.h>
#define GUANACO_RAND rand
#define GUANACO_RAND_MAX RAND_MAX
#endif  // GUANACO_RAND

#ifndef GUANACO_ASSERT
#include <assert.h>
#define GUANACO_ASSERT assert
#endif // GUANACO_ASSERT

#ifndef GUANACO_EXP
#include <math.h>
#define GUANACO_EXP expf
#endif  // GUANACO_EXP
