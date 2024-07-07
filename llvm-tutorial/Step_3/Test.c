#include <stdio.h>

void init(float* arr,float* brr,float* crr, size_t n)
{
     for(size_t i=0; i<n;i++)
     {
          arr[i]=n-i;
          brr[i]=n+n+i;
          crr[i]=i+n;
     }
}

void VectorAdd(float*arr,float*brr,float*crr,size_t n)
{
     for(size_t i=0; i<n ; i++)
     {
          crr[i] = (arr[i]+brr[i])/arr[i];
     }
}

void FuseAddMul(float*arr,float*brr,float*crr,size_t n)
{
     for(size_t i=0; i<n; i++)
     {
          crr[i]=arr[i]*brr[i]+crr[i];
     }
}

int main()
{
     float arr[10];
     float brr[10];
     float crr[10];

     init(arr,brr,crr,10);
     VectorAdd(arr,brr,crr,10);
     FuseAddMul(arr,brr,crr,10);

     for(size_t i=0; i<10; i++)
     {
          printf("%f\n",crr[i]);
     }
}

