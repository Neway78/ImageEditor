__kernel void laplacian(__read_only image2d_t gaussianDown,
                        __read_only image2d_t gaussianUp,       
                        __write_only image2d_t outputLaplacian,
                        sampler_t sampler)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 value = convert_float4(read_imageui(gaussianUp, sampler, (int2)(x,y))) - 
                   convert_float4(read_imageui(gaussianDown, sampler, (int2)(x,y)));

    value = 2*value + 128;
    value.w = 255;

    write_imageui(outputLaplacian, (int2)(x,y), convert_uint4(value));
}