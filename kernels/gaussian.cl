__kernel void gaussian(__read_only image2d_t inputImage,
                       __write_only image2d_t outputImage,
                       __constant float *filter,
                       sampler_t sampler)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 value = {0,0,0,0};

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2 ; j++) {
            value += filter[5*(i+2)+(j+2)] * convert_float4(read_imageui(inputImage, sampler, (int2)(x+i,y+j))); 
        }
    }
    write_imageui(outputImage, (int2)(x,y), convert_uint4(value));
}