#pragma once

/* Ray-Triangle Intersection Test Routines          */
/* Different optimizations of my and Ben Trumbore's */
/* code from journals of graphics tools (JGT)       */
/* http://www.acm.org/jgt/                          */
/* by Tomas Moller, May 2000                        */

/*
Copyright 2020 Tomas Akenine-Möller

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <math.h>
#include <cuda_runtime.h>
#include "commonCUDA.cuh"
#pragma once

#define EPSILON 0.000001
#define CROSS(dest, v1, v2)                                                                                                      \
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1];                                                                                       \
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2];                                                                                       \
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
#define DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
#define SUB(dest, v1, v2)                                                                                                        \
  dest[0] = v1[0] - v2[0];                                                                                                       \
  dest[1] = v1[1] - v2[1];                                                                                                       \
  dest[2] = v1[2] - v2[2];
HD int GPUintersect_triangle(double orig[3], double dir[3], double vert0[3], double vert1[3], double vert2[3], double *t, double *u,
                       double *v);
HD int GPUintersect_triangle1(double orig[3], double dir[3], double vert0[3], double vert1[3], double vert2[3], double *t, double *u,
                        double *v);
HD int GPUintersect_triangle2(double orig[3], double dir[3], double vert0[3], double vert1[3], double vert2[3], double *t, double *u,
                        double *v);
HD int GPUintersect_triangle3(double orig[3], double dir[3], double vert0[3], double vert1[3], double vert2[3], double *t, double *u,
                        double *v);
