CXX = g++ 
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3

NVCXX = nvcc 
NVCXXFLAGS = -g --ptxas-options=-v -std=c++11  -O3


CUDALIB = /usr/local/cuda/lib64
CUDAINC = /usr/local/cuda/include
SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(wildcard $(OBJDIR)/*.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all:  bin/reduce

bin:
	mkdir -p bin


bin/reduce: $(OBJS) $(CUOBJS) 
	mkdir -p bin
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
			    @echo "Compiled "$<" successfully!"

1: bin/reduce
	./bin/reduce /data-reduction/input_64.txt /data-reduction/output_64.txt

2: bin/reduce
	./bin/reduce /data-reduction/input_16384.txt /data-reduction/output_16384.txt

3: bin/reduce
	./bin/reduce /data-reduction/input_16777216.txt /data-reduction/output_16777216.txt

.PHONY:	test clean

$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
		mkdir -p cuobj
	    @echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"

clean: 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin/*

