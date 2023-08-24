#include <math.h>
#include "mass_matrix_func_c.h"

void evaluate(
              double input_0[4],
              double input_1[4],
              double input_2[6],
              double output_0[16]
             )
{

    double pydy_0 = input_2[4] * cos(input_2[2]);
    double pydy_1 = input_2[4] * sin(input_2[2]);

    output_0[0] = input_2[3] + input_2[4];
    output_0[1] = pydy_0;
    output_0[2] = pydy_1;
    output_0[3] = 0;
    output_0[4] = pydy_0;
    output_0[5] = input_2[4];
    output_0[6] = 0;
    output_0[7] = 0;
    output_0[8] = pydy_1;
    output_0[9] = 0;
    output_0[10] = input_2[4];
    output_0[11] = 0;
    output_0[12] = 0;
    output_0[13] = 0;
    output_0[14] = 0;
    output_0[15] = input_2[5];

}