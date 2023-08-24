#include <math.h>
#include "distance_to_axle_func_c.h"

void evaluate(
              double input_0[4],
              double input_1[6],
              double output_0[1]
             )
{

    

    output_0[0] = (-input_1[1] + input_0[1])*cos(input_1[2]) + input_0[2] *
    sin(input_1[2]) + input_0[0];

}