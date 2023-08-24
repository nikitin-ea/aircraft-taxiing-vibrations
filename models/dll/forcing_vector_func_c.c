#include <math.h>
#include "forcing_vector_func_c.h"

void evaluate(
              double input_0[4],
              double input_1[4],
              double input_2[6],
              double input_3[6],
              double output_0[4]
             )
{

    double pydy_0 = input_3[0] * input_3[4];
    double pydy_1 = sin(input_3[2]);
    double pydy_2 = cos(input_3[2]);

    output_0[0] = input_2[5] + input_2[3] - input_3[0] * input_3[3] - pydy_0;
    output_0[1] = -input_2[0] - input_2[2] * pydy_1 + input_2[3] * pydy_2 -
    pydy_0*pydy_2;
    output_0[2] = -input_2[1] + input_2[2] * pydy_2 + input_2[3] * pydy_1 -
    pydy_0*pydy_1;
    output_0[3] = input_2[4];

}