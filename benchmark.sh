mix run benchmarks/gpuraytracer.ex 256 20 1
mix run benchmarks/cpuraytracer.ex 256 20 1

mix run benchmarks/gpudp.ex 33792 1
mix run benchmarks/cpudp.ex 33792 1



nvcc -o c_implementations/craytracer c_implementations/craytracer.cu
./c_implementations/