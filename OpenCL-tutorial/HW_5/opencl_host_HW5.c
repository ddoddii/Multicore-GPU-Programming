#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "../common/time_check.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CL_ERR_TO_STR(err) case err: return #err

char const *
clGetErrorString(cl_int const err)
{
	switch(err)
	{
		CL_ERR_TO_STR(CL_SUCCESS);
		CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
		CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
		CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
		CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_MAP_FAILURE);
		CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
		CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
		CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
		CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_INVALID_VALUE);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
		CL_ERR_TO_STR(CL_INVALID_PLATFORM);
		CL_ERR_TO_STR(CL_INVALID_DEVICE);
		CL_ERR_TO_STR(CL_INVALID_CONTEXT);
		CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
		CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
		CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
		CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
		CL_ERR_TO_STR(CL_INVALID_SAMPLER);
		CL_ERR_TO_STR(CL_INVALID_BINARY);
		CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
		CL_ERR_TO_STR(CL_INVALID_KERNEL);
		CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
		CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
		CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
		CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
		CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
		CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
		CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
		CL_ERR_TO_STR(CL_INVALID_EVENT);
		CL_ERR_TO_STR(CL_INVALID_OPERATION);
		CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
		CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
		CL_ERR_TO_STR(CL_INVALID_PROPERTY);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);

			default:
		return "UNKNOWN ERROR CODE";
	}
}

cl_int
cl_assert(cl_int const code, char const * const file, int const line, bool const abort)
{
	if (code != CL_SUCCESS)
	{
		char const * const err_str = clGetErrorString(code);

		fprintf(stderr, "\"%s\", line %d: cl_assert (%d) = \"%s\"", file, line, code, err_str);

		if (abort)
		{
			exit(code);
		}
	}

	return code;
}

#define MAX_SOURCE_SIZE (0x100000)

#define cl(...)		cl_assert((cl##__VA_ARGS__), __FILE__, __LINE__, true);
#define cl_ok(err)	cl_assert(err, __FILE__, __LINE__, true);

int main(void) {

	printf("HW5: SGEMM with vectorization \n");

	// Create the two input vectors
	int GEMM_M = 2048;
	int GEMM_N = 1536;
	int GEMM_K = 1024;

	float *A = (float *)malloc(sizeof(float) * GEMM_M * GEMM_K);
	float *B = (float *)malloc(sizeof(float) * GEMM_K * GEMM_N);

	int i, j, k;

	for(i = 0; i < GEMM_M; i++) {
		for(j = 0; j < GEMM_K; j++) {
			A[i * GEMM_K + j] = (rand() / (float)RAND_MAX) * (0.5 - 0) + 0.5;
		}
	}
	
	for(i = 0; i < GEMM_K; i++) {
		for(j = 0; j < GEMM_N; j++) {
			B[i * GEMM_N + j] = (rand() / (float)RAND_MAX) * (0.5 - 0) + 0.5;
		}
	}

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("matmul_HW5.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;   
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	cl_ok(ret);

	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
			&device_id, &ret_num_devices);
	cl_ok(ret);

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	cl_ok(ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	cl_ok(ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
			GEMM_M * GEMM_K * sizeof(float), NULL, &ret);
	cl_ok(ret);

	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			GEMM_K * GEMM_N * sizeof(float), NULL, &ret);
	cl_ok(ret);

	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			GEMM_M * GEMM_N * sizeof(float), NULL, &ret);
	cl_ok(ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			GEMM_M * GEMM_K * sizeof(float), A, 0, NULL, NULL);
	cl_ok(ret);

	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			GEMM_K * GEMM_N * sizeof(float), B, 0, NULL, NULL);
	cl_ok(ret);

	clFinish(command_queue);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, 
			(const char **)&source_str, (const size_t *)&source_size, &ret);
	cl_ok(ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	cl_ok(ret);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "matmul_HW5", &ret);
	cl_ok(ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(int), &GEMM_M);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 1, sizeof(int), &GEMM_N);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 2, sizeof(int), &GEMM_K);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&a_mem_obj);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&b_mem_obj);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&c_mem_obj);
	cl_ok(ret);

	// Execute the OpenCL kernel on the list
	size_t global_item_size[2] = {GEMM_M, GEMM_N};

	double start_time, end_time;

	start_time = get_time();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
			global_item_size, NULL, 0, NULL, NULL);
	cl_ok(ret);

	end_time = get_time();
	clFinish(command_queue);

	// Read the memory buffer C on the device to the local variable C
	float *C = (float *)malloc(sizeof(float) * GEMM_M * GEMM_N);
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
			GEMM_M * GEMM_N * sizeof(float), C, 0, NULL, NULL);
	cl_ok(ret);

	// Result check, compare match
	float *C_ref = (float *)malloc(sizeof(float) * GEMM_M * GEMM_N);
	for(i = 0; i < GEMM_M; i++) {
		for(j = 0; j < GEMM_N; j++) {
			float Csub = 0.0f;
			for(k = 0; k < GEMM_K; k++) {
				Csub += A[i * GEMM_K + k] * B[j + GEMM_N * k];
			}
			C_ref[i * GEMM_N + j] = Csub;

		}
	}

	int res_count = 0;
	for(i = 0; i < GEMM_M * GEMM_N; i++) {
		float cmp_abs = C_ref[i] - C[i] > 0 ? C_ref[i] - C[i] : (C_ref[i] - C[i]) * (float)(-1);
		
		if( cmp_abs / C_ref[i] >= 0.001f )
			res_count += 1;
	}

	float res_check = (res_count / (float)(GEMM_M * GEMM_N)) * 100.0f;
	printf("Performance: %.9lf sec, result: %s (Over 0.0001 percentage error ratio: %.2f) \n", end_time - start_time, res_check <= 5.0f ? "PASSED" : "FAILED", res_check);


	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	free(C_ref);
	return 0;
}
