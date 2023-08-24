import numpy as np
cimport numpy as np
cimport cython

cdef extern from "mass_matrix_func_c.h":
    void evaluate(
                  double* input_0,
                  double* input_1,
                  double* input_2,
                  double* output_0
                 )

@cython.boundscheck(False)
@cython.wraparound(False)
def eval(
         np.ndarray[np.double_t, ndim=1, mode='c'] input_0,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_1,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_2,
         np.ndarray[np.double_t, ndim=1, mode='c'] output_0
        ):

    evaluate(
             <double*> input_0.data,
             <double*> input_1.data,
             <double*> input_2.data,
             <double*> output_0.data
            )

    return (
            output_0.reshape(4, 4)
           )