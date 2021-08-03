#include <fstream>
#include <iostream>

#include "kernel.h"

Kernel::Kernel(cl::Context &context, std::vector<cl::Device> &devices, 
               const char *path, const char *kernelName)
{
    std::ifstream sourceFile(path);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),
                           (std::istreambuf_iterator<char>()));
    _source = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), 
                                   sourceCode.length() + 1));

    _program = cl::Program(context, _source);
    
    try 
    {
        _program.build(devices);
    }
    catch (cl::Error &e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            for (cl::Device dev : devices) {
                // Check the build status
                cl_build_status status = _program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name     = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                          << buildlog << std::endl;
            }
        } else {
            throw e;
        }
    }

    try 
    {
        _kernel = cl::Kernel(_program, kernelName);
    }
    catch (cl::Error &error)
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}