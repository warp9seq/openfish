CC = gcc
AR = ar
CPPFLAGS +=	-I include/
CFLAGS += -g -Wall -O2
# auto-generate header dependencies so editing a .h (e.g. beam_search_hip.h) rebuilds dependent objects
DEPFLAGS = -MMD -MP
LDFLAGS += $(LIBS) -lz -lm -lpthread
BUILD_DIR = lib

# change the tool name to what you want
BINARY = openfish

STATICLIB = $(BUILD_DIR)/libopenfish.a

OBJ = $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/decode_cpu.o \
	  $(BUILD_DIR)/openfish.o \
	  $(BUILD_DIR)/beam_search.o \

GPU_LIB =

# add more objects here if needed
VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef cuda
	CUDA_ROOT ?= /usr/local/cuda
    CUDA_LIB ?= $(CUDA_ROOT)/lib64
    CUDA_OBJ += $(BUILD_DIR)/decode_cuda.o
    NVCC ?= $(CUDA_ROOT)/bin/nvcc
    CUDA_CFLAGS += -g -O2 -lineinfo $(CUDA_ARCH) -Xcompiler -Wall
    CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart_static -lrt -ldl
    GPU_LIB = $(BUILD_DIR)/cuda.a
    CPPFLAGS += -DHAVE_CUDA=1
    MAIN_CC = $(NVCC)
    MAIN_CFLAGS = -x cu $(CUDA_CFLAGS)
else ifdef rocm
	ROCM_ROOT ?= /opt/rocm
	ROCM_LIB ?= $(ROCM_ROOT)/lib
	HIPCC ?= $(ROCM_ROOT)/bin/hipcc
	ROCM_CFLAGS += -g -Wall $(ROCM_ARCH)
# 	ifneq (,$(findstring gfx1150,$(ROCM_ARCH)))
# 		ROCM_CFLAGS += -D__AMDGCN_WAVEFRONT_SIZE=32
# 	endif
	ROCM_OBJ += $(BUILD_DIR)/decode_hip.o
	GPU_LIB = $(BUILD_DIR)/hip_code.a
	ROCM_LDFLAGS = -L$(ROCM_LIB) -lamdhip64 -lrt -ldl
	CPPFLAGS += -DHAVE_ROCM=1
	MAIN_CC = $(HIPCC)
	MAIN_CFLAGS = -x hip $(ROCM_CFLAGS) -fPIC
else
	GPU_LIB = $(BUILD_DIR)/cpu_decoy.a
	MAIN_CC = $(CC)
	MAIN_CFLAGS = $(CFLAGS)
endif

ifdef bench
	CPPFLAGS += -DBENCH=1
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
endif

.PHONY: clean distclean test

$(BINARY): $(BUILD_DIR)/main.o $(STATICLIB)
	$(CC) $(CFLAGS) $(BUILD_DIR)/main.o $(STATICLIB) $(LDFLAGS) $(CUDA_LDFLAGS) $(ROCM_LDFLAGS) -o $@

$(STATICLIB): $(OBJ) $(GPU_LIB)
	cp $(GPU_LIB) $@
	$(AR) rcs $@ $(OBJ)

$(BUILD_DIR)/main.o: src/main.c include/openfish/openfish.h
	$(MAIN_CC) $(MAIN_CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/misc.o: src/misc.c src/misc.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/error.o: src/error.c include/openfish/openfish_error.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/decode_cpu.o: src/decode_cpu.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/beam_search.o: src/beam_search.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/openfish.o: src/openfish.c include/openfish/openfish.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

# cpu decoy: an empty static archive that $(STATICLIB) copies as its base before
# adding $(OBJ). BSD/macOS ar cannot create an archive with no members, so write
# the archive magic directly — both GNU and BSD ar accept this as a valid empty archive.
$(BUILD_DIR)/cpu_decoy.a:
	rm -f $@
	printf '!<arch>\n' > $@

# cuda
$(BUILD_DIR)/cuda.a: $(CUDA_OBJ)
	$(AR) rcs $@ $^

$(BUILD_DIR)/decode_cuda.o: src/decode_cuda.c
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

# hip
$(BUILD_DIR)/hip_code.a: $(ROCM_OBJ)
	$(HIPCC) $(ROCM_CFLAGS) --emit-static-lib -fPIC --hip-link $^ -o $@

$(BUILD_DIR)/decode_hip.o: src/decode_hip.c
	$(HIPCC) -x hip $(ROCM_CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -fPIC -c $< -o $@

# pull in auto-generated header dependencies (.d files emitted by -MMD)
-include $(BUILD_DIR)/*.d

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BINARY) $(BUILD_DIR)/* autom4te.cache

# make test with run a simple test
test: $(BINARY)
	./test/test.sh