#########################
# Variables
#########################
NVCC=/usr/local/cuda/bin/nvcc
BUILD=build
SRC=src
FLAGS=-std=c++11 -O3 $(shell pkg-config --cflags opencv4)
LIBS=$(shell pkg-config --libs opencv4)

#########################
# Main
#########################

run : predict
	./predict /vgg_data/cifar10/test_batch.bin 0 128 tmp/cifar10_test_%d_%s.bmp /vgg_data/vgg_weight/values_vgg.txt

predict: $(BUILD)/main.o $(BUILD)/util.o $(BUILD)/vgg16_cpu.o $(BUILD)/vgg16_cuda.o $(BUILD)/vgg16.o 
	$(NVCC) $(FLAGS) $(LIBS) -o $@ $^

$(BUILD)/main.o: $(SRC)/main.cpp $(BUILD)/util.o $(BUILD)/vgg16.o $(BUILD)/vgg16_cpu.o $(BUILD)/vgg16_cuda.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $< 

$(BUILD)/util.o: $(SRC)/util.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $< 

$(BUILD)/vgg16_cpu.o: $(SRC)/vgg16_cpu.cpp $(BUILD)/vgg16.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

$(BUILD)/vgg16_cuda.o: $(SRC)/vgg16_cuda.cu $(BUILD)/vgg16.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

$(BUILD)/vgg16.o: $(SRC)/vgg16.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

clean:
	rm -rf predict result/* tmp/*.log tmp/*.bmp $(BUILD)/*.o

