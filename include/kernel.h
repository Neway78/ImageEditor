#ifndef KERNEL_H
#define KERNEL_H

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

/**
 * \brief Wrapper Class for an OpenCL Kernel
 */
class Kernel
{
public:
    /**
     * \param _source OpenCL Source file object 
     */
    cl::Program::Sources    _source;

    /**
     * \param _program OpenCL Program built with the kernel source
     */
    cl::Program             _program;

    /**
     * \param _kernel OpenCL kernel object
     */
    cl::Kernel              _kernel;


    /**
     * \brief Constructor for an Open Kernel Object
     *
     * \param context OpenCL context
     * \param devices list of devices for the chosen platform
     * \param path source file path of the OpenCL Kernel file (.cl)
     * \param kernelName name of the kernel in the .cl file
     */
    Kernel(cl::Context &context, std::vector<cl::Device> &devices, 
           const char *path, const char *kernelName);
};

#endif
