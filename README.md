# HIPGEMM

Companion code for [*Deep Dive Into 4-Wave Interleave FP8 GEMM*](...).

## Building and Testing
> Note: the code has been tested on AMD Instinct MI355X using ROCm 7.2.2 (see the attached devcontainer file). For the specific instructions used, a CDNA4-capable GPU is required to run this code.

Test:
```
make test
````

Benchmark:
```bash
make bench
````

