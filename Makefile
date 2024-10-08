CC       = gcc
CXX		 = g++

CPPFLAGS +=	-I src/ -I include/
CFLAGS	+= 	-g -Wall -O2
CXXFLAGS   += -g -Wall -O2 -std=c++14
LDFLAGS  += $(LIBS) -lz -lm -lpthread -lstdc++fs
BUILD_DIR = build

# https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
ifeq ($(cxx11_abi),) #  cxx11_abi not defined
CXXFLAGS		+= -D_GLIBCXX_USE_CXX11_ABI=0
endif

# change the tool name to what you want
BINARY = openfish

OBJ = $(BUILD_DIR)/main.o \
	  $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/decode_cpu.o \
	  $(BUILD_DIR)/openfish.o \
	  $(BUILD_DIR)/beam_search.o \

# add more objects here if needed

VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef cuda
	CUDA_ROOT = /usr/local/cuda
    CUDA_LIB ?= $(CUDA_ROOT)/lib64
    CUDA_OBJ += $(BUILD_DIR)/decode_cuda.o $(BUILD_DIR)/beam_search_cuda.o $(BUILD_DIR)/scan_cuda.o
    NVCC ?= nvcc
    CUDA_CFLAGS += -g -O2 -std=c++11 -lineinfo $(CUDA_ARCH) -Xcompiler -Wall
    CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart_static -lrt -ldl
    OBJ += $(BUILD_DIR)/cuda_code.o $(CUDA_OBJ)
    CPPFLAGS += -DHAVE_CUDA=1
else ifdef rocm
	ROCM_ROOT = /opt/rocm
	HIP_INCLUDE_DIR = $(ROCM_ROOT)/include
	HIPCXX ?= $(ROCM_ROOT)/bin/hipcc
	HIP_CFLAGS += -g -std=c++17 -Wall -Wextra
	HIP_OBJ += $(BUILD_DIR)/beam_search_hip.o $(BUILD_DIR)/scan_hip.o
	HIP_LDFLAGS = -lrt -ldl --hip-link
	OBJ += $(BUILD_DIR)/hip_code.o $(HIP_OBJ) $(BUILD_DIR)/decode_hip.o
	CPPFLAGS += -DHAVE_HIP=1
endif

ifdef bench
	CPPFLAGS += -DBENCH=1
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
endif

.PHONY: clean distclean test

$(BINARY): $(OBJ)
	$(CXX) $(CFLAGS) $(OBJ) $(LDFLAGS) $(CUDA_LDFLAGS) $(HIP_LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp include/openfish/openfish.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/misc.o: src/misc.cpp src/misc.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/error.o: src/error.cpp src/error.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/signal_prep.o: src/signal_prep.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/decode_cpu.o: src/decode_cpu.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/beam_search.o: src/beam_search.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/openfish.o: src/openfish.cpp include/openfish/openfish.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/decode_hip.o: src/decode_hip.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# cuda
$(BUILD_DIR)/cuda_code.o: $(CUDA_OBJ)
	$(NVCC) $(CUDA_CFLAGS) -dlink $^ -o $@

$(BUILD_DIR)/decode_cuda.o: src/decode_cuda.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -rdc=true -c $< -o $@

$(BUILD_DIR)/beam_search_cuda.o: src/beam_search_cuda.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -rdc=true -c $< -o $@

$(BUILD_DIR)/scan_cuda.o: src/scan_cuda.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -rdc=true -c $< -o $@

# hip
$(BUILD_DIR)/hip_code.o: $(HIP_OBJ)
	$(HIPCXX) $(HIP_CFLAGS) -dlink $^ -o $@

$(BUILD_DIR)/beam_search_hip.o: src/beam_search_hip.hip
	$(HIPCXX) -x hip $(HIP_CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/scan_hip.o: src/scan_hip.hip
	$(HIPCXX) -x hip $(HIP_CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BUILD_DIR)/* autom4te.cache
