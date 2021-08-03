#ifndef OPENCL_PROGRAM_H
#define OPENCL_PROGRAM_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

/**
 * \brief Wrapper class of an OpenCL Program
 */
class OpenCLProgram
{
public:
    /**
     * \param _platform OpenCL Platform
     */
    cl::Platform            _platform;

    /**
     * \param _devices list of devices for this platform
     */
    std::vector<cl::Device> _devices;

    /**
     * \param _context OpenCL Context
     */
    cl::Context             _context;

    /**
     * \param _queue OpenCL Command Queue
     */
    cl::CommandQueue        _queue;

    OpenCLProgram();

    void initProgram();
};

#endif
