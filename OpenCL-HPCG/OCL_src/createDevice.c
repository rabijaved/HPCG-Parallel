#include "createDevice.h"

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

// Find a GPU or CPU for platform 
cl_device_id create_device(int plat_id, int dev_id) {


	if(plat_id == -1 && dev_id == -1){

		printf("Correct usage is:\n");
		printf("\t./c_cwt <platform no.> <device no.>\n");
		//start by getting the count of available platforms
		cl_uint number_of_platforms = 0;
		clGetPlatformIDs((cl_uint)NULL,       
		NULL,                 
		&number_of_platforms);
		if(number_of_platforms==0){
			printf("Error: No platforms found!\n");
			printf("\tIs an OpenCL driver installed?");
			return NULL;
		}

		
		cl_platform_id* my_platforms =
		(cl_platform_id*)malloc(sizeof(cl_platform_id)*
		number_of_platforms);
		clGetPlatformIDs(number_of_platforms,
		my_platforms,       
		NULL);             

		printf("\n");
		printf("Your system platform id(s) are:\n");
		for(size_t i = 0; i < number_of_platforms; i++){
			printf("\tplatform no. %lu\n",i);

			//print it's name
			size_t total_buffer_length = 1024;
			size_t length_of_buffer_used = 0;
			char my_platform_name[total_buffer_length];
			clGetPlatformInfo(my_platforms[i],       
			CL_PLATFORM_NAME,      
			total_buffer_length,   
			&my_platform_name,     
			&length_of_buffer_used);
			printf("\t\tname:\t\t%*.*s\n",(int)length_of_buffer_used,
			(int)length_of_buffer_used,my_platform_name);

			//print the vendor
			char my_platform_vendor[total_buffer_length];
			length_of_buffer_used = 0;
			clGetPlatformInfo(my_platforms[i],        
			CL_PLATFORM_VENDOR,     
			total_buffer_length,    
			&my_platform_vendor,    
			&length_of_buffer_used);
			printf("\t\tvendor:\t\t%*.*s\n",(int)length_of_buffer_used,
			(int)length_of_buffer_used,my_platform_vendor);

			//print the profile
			char my_platform_profile[total_buffer_length];
			length_of_buffer_used = 0;
			clGetPlatformInfo(my_platforms[i],        
			CL_PLATFORM_PROFILE,    
			total_buffer_length,    
			&my_platform_profile,   
			&length_of_buffer_used);
			printf("\t\tprofile:\t%*.*s\n",(int)length_of_buffer_used,
			(int)length_of_buffer_used,my_platform_profile);

			//print the extensions
			char my_platform_extensions[total_buffer_length];
			length_of_buffer_used = 0;
			clGetPlatformInfo(my_platforms[i],        
			CL_PLATFORM_EXTENSIONS, 
			total_buffer_length,    
			&my_platform_extensions,
			&length_of_buffer_used);
			printf("\t\textensions:\t%*.*s\n",(int)length_of_buffer_used,
			(int)length_of_buffer_used,my_platform_extensions);

			cl_uint number_of_devices = 0;
			clGetDeviceIDs(my_platforms[i],    
			CL_DEVICE_TYPE_ALL, 
			(cl_uint)NULL,      
			NULL,               
			&number_of_devices);
			if(number_of_devices==0){
				printf("Error: No devices found for this platform!\n");
				return NULL;
			}
			printf("\n\t\twith device id(s):\n");

			//get all those platforms
			cl_device_id* my_devices =
			(cl_device_id*)malloc(sizeof(cl_device_id)*number_of_devices);
			clGetDeviceIDs(my_platforms[i],    
			CL_DEVICE_TYPE_ALL, 
			number_of_devices,  
			my_devices,         
			NULL);              
			
			for(size_t j = 0; j < number_of_devices; j++){
				printf("\t\tdevice no. %lu\n",j);

				//print the name
				char my_device_name[total_buffer_length];
				length_of_buffer_used = 0;
				clGetDeviceInfo(my_devices[i],          
				CL_DEVICE_NAME,         
				total_buffer_length,    
				&my_device_name,        
				&length_of_buffer_used);
				printf("\t\t\tname:\t\t%*.*s\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_device_name);

				//print the vendor
				char my_device_vendor[total_buffer_length];
				length_of_buffer_used = 0;
				clGetDeviceInfo(my_devices[i],          
				CL_DEVICE_VENDOR,       
				total_buffer_length,    
				&my_device_vendor,      
				&length_of_buffer_used);
				printf("\t\t\tvendor:\t\t%*.*s\n\n",(int)length_of_buffer_used,
				(int)length_of_buffer_used,my_device_vendor);
			}
			printf("\n");
			free(my_devices);
		}

		free(my_platforms);


		printf("Enter platform id: ");
		scanf("%d", &plat_id);


		printf("Enter device id: ");
		scanf("%d", &dev_id);

	}
	//get arguments for device
	size_t target_platform_id = plat_id;
	size_t target_device_id = dev_id;

	cl_platform_id my_platform;
	cl_int error_id;
	cl_uint number_of_platforms = 0;
	clGetPlatformIDs((cl_uint)NULL,        
	NULL,                 
	&number_of_platforms);
	if(number_of_platforms==0){
		printf("Error: No platforms found!\n");
		printf("\tIs an OpenCL driver installed?");
		return NULL;
	}else if(target_platform_id >= number_of_platforms){
		printf("Error: incorrect platform id given!\n");
		printf("\t%lu was provided but only %i platforms found.",
		target_platform_id,
		number_of_platforms);
		return NULL;
	}

	cl_platform_id* my_platforms =
	(cl_platform_id*)malloc(sizeof(cl_platform_id)*number_of_platforms);

	error_id = clGetPlatformIDs(number_of_platforms,
	my_platforms, 
	NULL);        

	if(error_id != CL_SUCCESS){
		printf("there was an error getting the platform!\n");
		printf("\tdoes platform no. %lu exist?",target_platform_id);
		return NULL;
	}

	my_platform = my_platforms[target_platform_id];

	//now get the target device
	cl_device_id my_device;
	cl_uint number_of_devices = 0;
	clGetDeviceIDs(my_platform,        
	CL_DEVICE_TYPE_ALL, 
	(cl_uint)NULL,      
	NULL,               
	&number_of_devices);
	if(number_of_devices==0){
		printf("Error: No devices found for this platform!\n");
		return NULL;
	}else if(target_device_id >= number_of_devices){
		printf("Error: incorrect device id given!\n");
		printf("\t%lu was provided but only %i devices found.",
		target_device_id,
		number_of_devices);
		return NULL;
	}

	cl_platform_id* my_devices = (cl_platform_id*)malloc(sizeof(cl_device_id)*
	number_of_devices);

	error_id = clGetDeviceIDs(my_platform,                  
	CL_DEVICE_TYPE_ALL,           
	number_of_devices,            
	(cl_device_id*)my_devices,    
	NULL);                        
	if(error_id != CL_SUCCESS){
		printf("there was an error getting the device!\n");
		printf("\tdoes device no. %lu exist?",target_device_id);
		return NULL;
	}

	my_device = (cl_device_id)my_devices[target_device_id];


	return my_device;
}
