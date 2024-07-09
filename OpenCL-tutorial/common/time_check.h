#include <sys/time.h>
#include <unistd.h>

inline double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)1.0e-6 * tv.tv_usec;
}
