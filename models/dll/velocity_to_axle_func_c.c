#include <math.h>
#include "velocity_to_axle_func_c.h"

void evaluate(
              double input_0[4],
              double input_1[4],
              double input_2[6],
              double output_0[1]
             )
{

    

    output_0[0] = input_1[0] + input_1[1] * cos(input_2[2]) + input_1[2] *
    sin(input_2[2]);

}