HIPCC		=	hipcc
HIPCCFLAGS	=	-std=c++20 -Rpass-analysis=kernel-resource-usage --offload-arch=gfx950 -Wall -Wextra -fno-exceptions -fno-rtti -fopenmp
LDFLAGS		=	-lm

ROCM		=	/opt/rocm

HIPCCFLAGS	+=	-I$(ROCM)/include -D__HIP_PLATFORM_AMD__
LDFLAGS		+=	-L$(ROCM)/lib -lamdhip64


.PHONY: clean test bench

clean:
	rm -rf *.o test bench

test: test.o
	$(HIPCC) $(HIPCCFLAGS) $< -o $@ $(LDFLAGS)
	@./test

test.o: test.cpp
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@
	
bench: bench.o
	$(HIPCC) $(HIPCCFLAGS) $< -o $@ $(LDFLAGS)
	@./bench

bench.o: bench.cpp
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@
