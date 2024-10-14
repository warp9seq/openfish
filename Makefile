CC       = gcc
AR = ar
CPPFLAGS +=	-I include/
CFLAGS	+= 	-g -Wall -O2
LDFLAGS  += $(LIBS) -lz -lm -lpthread
BUILD_DIR = lib

# change the tool name to what you want
BINARY = openfish
STATICLIB = $(BUILD_DIR)/libopenfish.a

OBJ = $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/decode_cpu.o \
	  $(BUILD_DIR)/openfish.o \
	  $(BUILD_DIR)/beam_search.o \

# add more objects here if needed
VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef cuda
	CUDA_ROOT = /usr/local/cuda
    CUDA_LIB ?= $(CUDA_ROOT)/lib64
    CUDA_OBJ += $(BUILD_DIR)/decode_cuda.o $(BUILD_DIR)/beam_search_cuda.o $(BUILD_DIR)/scan_cuda.o
    NVCC ?= nvcc
    CUDA_CFLAGS += -g -O2 -lineinfo $(CUDA_ARCH) -Xcompiler -Wall
    CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart_static -lrt -ldl
    OBJ += $(BUILD_DIR)/cuda_code.o $(CUDA_OBJ)
    CPPFLAGS += -DHAVE_CUDA=1
else ifdef rocm
	ROCM_ROOT = /opt/rocm
	HIP_INCLUDE_DIR = $(ROCM_ROOT)/include
	HIP_LIB ?= $(ROCM_ROOT)/lib
	HIPCXX ?= $(ROCM_ROOT)/bin/hipcc
	HIP_CFLAGS += -g -Wall -Wextra
	HIP_OBJ += $(BUILD_DIR)/decode_hip.o $(BUILD_DIR)/beam_search_hip.o $(BUILD_DIR)/scan_hip.o
	HIP_LDFLAGS = -L$(HIP_LIB) -lamdhip64 -lrt -ldl
	OBJ += $(BUILD_DIR)/hip_code.a
	CPPFLAGS += -DHAVE_HIP=1
endif

ifdef bench
	CPPFLAGS += -DBENCH=1
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
endif

.PHONY: clean distclean test

$(BINARY): $(BUILD_DIR)/main.o $(STATICLIB)
	$(CC) $(CFLAGS) $(BUILD_DIR)/main.o $(STATICLIB) $(LDFLAGS) $(CUDA_LDFLAGS) $(HIP_LDFLAGS) -o $@

$(STATICLIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

$(BUILD_DIR)/main.o: src/main.c include/openfish/openfish.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/misc.o: src/misc.c src/misc.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/error.o: src/error.c src/error.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/signal_prep.o: src/signal_prep.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/decode_cpu.o: src/decode_cpu.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/beam_search.o: src/beam_search.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/openfish.o: src/openfish.c include/openfish/openfish.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

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
$(BUILD_DIR)/hip_code.a: $(HIP_OBJ)
	$(HIPCXX) $(HIP_CFLAGS) --emit-static-lib -fPIC -fgpu-rdc $^ -o $@

$(BUILD_DIR)/beam_search_hip.o: src/beam_search_hip.hip
	$(HIPCXX) -x hip $(HIP_CFLAGS) $(CPPFLAGS) -fgpu-rdc -fPIC -c $< -o $@

$(BUILD_DIR)/scan_hip.o: src/scan_hip.hip
	$(HIPCXX) -x hip $(HIP_CFLAGS) $(CPPFLAGS) -fgpu-rdc -fPIC -c $< -o $@

$(BUILD_DIR)/decode_hip.o: src/decode_hip.hip
	$(HIPCXX) -x hip $(HIP_CFLAGS) $(CPPFLAGS) -fgpu-rdc -fPIC -c $< -o $@

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BUILD_DIR)/* autom4te.cache