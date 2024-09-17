CC       = cc
CXX		 = c++

CPPFLAGS +=	-I src/
CFLAGS	+= 	-g -Wall -O2
CXXFLAGS   += -g -Wall -O2  -std=c++14
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


# add more objects here if needed

VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
# ifdef cuda
#     CPPFLAGS += -DUSE_GPU=1
# 	OBJ += $(BUILD_DIR)/decode_gpu.o
# 	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cuda.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_cuda.so"
# 	LDFLAGS += -lrt -ldl
# else
# ifdef rocm
# 	CPPFLAGS += -DUSE_GPU=1
# 	OBJ += $(BUILD_DIR)/decode_gpu.o
# 	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_hip.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_hip.so"
# 	LDFLAGS += -lrt -ldl
# endif
# endif

.PHONY: clean distclean test

$(BINARY): $(OBJ)
	$(CXX) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/misc.o: src/misc.cpp src/misc.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/error.o: src/error.cpp src/error.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/signal_prep.o: src/signal_prep.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/decode_cpu.o: src/decode_cpu.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/decode_gpu.o: src/decode_gpu.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BUILD_DIR)/* autom4te.cache
