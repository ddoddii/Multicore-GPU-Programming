#include <stdio.h>

int add_inst_count = 0;

int main(void)
{
	int a = 3;

	for( int i=0; i<10; ++i )
	{
		a += i;
	}

	printf( "Number of ADD : %d\n", add_inst_count );

	return 0;
}
