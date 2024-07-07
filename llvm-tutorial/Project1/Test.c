#include <stdio.h>

int main(void)
{
	int a = 3, b = 4;
	float c = 3.0;

	a = a + 0;
	a = 0 + a;
	b = b * 1;
	b = 1 * b;
	c = c * 1.0f;

	return 0;
}
