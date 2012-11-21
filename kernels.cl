__kernel void take_neg_log(__global float *input, __global float *output)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int idx = y * width + x;
    output[idx] = -log(input[idx]);    
}
