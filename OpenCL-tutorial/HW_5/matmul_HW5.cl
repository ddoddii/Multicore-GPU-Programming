// HW_5
__kernel void matmul_HW5( 
		const int M,
		const int N,
		const int K,
		const __global float *A, 
		const __global float *B, 
		__global float *C)
{
	int tidx = get_global_id(0); // i
	int tidy = get_global_id(1); // j

	int vlen = 4;

	if (tidx < M && tidy < N)
	{
		float Csub = 0.0f;

		for(int k = 0; k < K; k += vlen) // k
		{
			float /* fill here */

			for (int l = 0; l < vlen; ++l)
			{
				/* fill here */
			}
		}

		C[tidx * N + tidy] = Csub;
	}
}


