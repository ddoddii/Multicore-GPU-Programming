#include <stdlib.h>
#include <stdio.h>
int Add(int*arr,int*brr,int*crr,size_t n)
{
     crr[n]= arr[n]+brr[n]+n;
     return crr[n];
}

int main()
{

     int N=0;
     printf("Input N (0~5)\n");
     scanf("%d",&N);
     int crr[5]={0,0,0,0,0};
     int arr[5]={1,2,3,4,5};
     int brr[5]={2,3,4,5,6};
     int result = Add(arr,brr,crr,N);
     printf("...(%d MUL %d) SUB %d = %d\n" ,arr[N],brr[N],N,result);
}
