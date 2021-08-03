#include <iostream>

#include "openclProgram.h"

OpenCLProgram::OpenCLProgram() {}

void OpenCLProgram::initProgram() 
{
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
    
        _platform = platforms[0];

        // ------------ BEGIN DEBUG --------------
        // std::string extensions, name, profile, vendor, version;
        // _platform.getInfo(CL_PLATFORM_EXTENSIONS, &extensions);
        // _platform.getInfo(CL_PLATFORM_NAME, &name);
        // _platform.getInfo(CL_PLATFORM_PROFILE, &profile);
        // _platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
        // _platform.getInfo(CL_PLATFORM_VERSION, &version);

        // std::cout << "\n>> PLATFORM" << std::endl;
        // std::cout << "\nExtensions: " << extensions << std::endl;
        // std::cout << "\nName: " << name << std::endl;
        // std::cout << "\nProfile: " << profile << std::endl;
        // std::cout << "\nVendor: " << vendor << std::endl;
        // std::cout << "\nVersion: " << version << std::endl;
        // ------------- END DEBUG ----------------

        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &_devices);

        // ------------ BEGIN DEBUG --------------
        // std::string built_in_kernels, d_extensions, d_name, opencl_c_version,
        //             d_profile, d_vendor, d_version, driver_version;
        // std::vector<size_t> max_work_item_sizes;
        // std::vector<cl_device_partition_property> partition_properties, partition_type;
        // _devices[0].getInfo(CL_DEVICE_BUILT_IN_KERNELS, &built_in_kernels);
        // _devices[0].getInfo(CL_DEVICE_EXTENSIONS, &d_extensions);
        // _devices[0].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
        // _devices[0].getInfo(CL_DEVICE_NAME, &d_name);
        // _devices[0].getInfo(CL_DEVICE_OPENCL_C_VERSION, &opencl_c_version);
        // _devices[0].getInfo(CL_DEVICE_PARTITION_PROPERTIES, &partition_properties);
        // _devices[0].getInfo(CL_DEVICE_PARTITION_TYPE, &partition_type);
        // _devices[0].getInfo(CL_DEVICE_PROFILE, &d_profile);
        // _devices[0].getInfo(CL_DEVICE_VENDOR, &d_vendor);
        // _devices[0].getInfo(CL_DEVICE_VERSION, &d_version);
        // _devices[0].getInfo(CL_DRIVER_VERSION, &driver_version);

        // std::cout << "\n>> DEVICE" << std::endl;
        // std::cout << "\nBuilt_in_kernels: " << built_in_kernels << std::endl;
        // std::cout << "\nd_Extensions: " << d_extensions << std::endl;
        // std::cout << "\nMax_work_item_sizes: ";
        // for (std::vector<size_t>::const_iterator i = max_work_item_sizes.begin(); i != max_work_item_sizes.end(); ++i)
        //     std::cout << *i << ' ';
        // std::cout << "\n\nd_Name: " << d_name << std::endl;
        // std::cout << "\nOpencl_c_version: " << opencl_c_version << std::endl;
        // std::cout << "\nPartition_properties:";
        // for (uint i = 0; i < partition_properties.size(); i++) {
        //     std::cout << " " << partition_properties[i];
        // }
        // std::cout << "\n\nPartition_type:";
        // for (uint i = 0; i < partition_type.size(); i++) {
        //     std::cout << " " << partition_type[i];
        // }
        // std::cout << "\n\nnd_Profile: " << d_profile << std::endl;
        // std::cout << "\nd_Vendor: " << d_vendor << std::endl;
        // std::cout << "\nd_Version: " << d_version << std::endl;
        // std::cout << "\nDriver_version: " << driver_version << std::endl;
        // ------------- END DEBUG ----------------

        _context = cl::Context(_devices);

        // ------------ BEGIN DEBUG --------------
        // std::vector<cl::Device> c_devices;
        // std::string c_name;
        // std::vector<cl_context_properties> c_properties;
        // _context.getInfo(CL_CONTEXT_DEVICES, &c_devices);
        // _context.getInfo(CL_CONTEXT_PROPERTIES, &c_properties);

        // std::cout << "\n>> CONTEXT" << std::endl;
        // std::cout << "\nDevices:";
        // for (uint i = 0; i < c_devices.size(); i++) {
        //     c_devices[i].getInfo(CL_DEVICE_NAME, &c_name);
        //     std::cout << " " << c_name;
        // }

        // std::vector<cl::ImageFormat> formats;
        // _context.getSupportedImageFormats(CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, &formats);

        // std::cout << "\nSupported Image Formats:";
        // for (uint i = 0; i < formats.size(); i++)
        //     std::cout << "\t{" << std::hex << formats[i].image_channel_data_type << ", " << std::hex << formats[i].image_channel_order << "}";

        // std::cout << "\n";
        // ------------- END DEBUG ----------------

        _queue = cl::CommandQueue(_context, _devices[0]);

    }
    catch (cl::Error &error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
