#include <stdio.h>

int main(void)
{
	int a = 3;
	int b = 4;

	a = a + 5;
	b = a + b;

	printf( "a(3+5) = %d, b(a+4) = %d\n", a, b );

	return 0;
}
