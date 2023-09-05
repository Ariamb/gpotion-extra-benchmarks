#!/bin/bash

N=30

# Compiling files
#sh compile.sh


for SIZE in 5000000 50000000 100000000; do
    echo $SIZE

    for i in $(seq 1 $N); do
        echo $i
    #sleep 1

	echo C GPU
        ./c_implementations/cdotproduct $SIZE $i
        #./c_implementations/craytracer $SIZE $i
    sleep 0.2
    
    echo Elixir CPU
        mix run benchmarks/cpudp.ex $SIZE $i
    sleep 0.2

    echo Matrex CPU
        mix run benchmarks/cpumatrexdp.ex $SIZE $i
    sleep 0.2

    echo Elixir GPU
        mix run benchmarks/gpudp.ex $SIZE $i
    done
    sleep 0.2
done



#mix run benchmarks/gpuraytracer.ex 256 20 1
#mix run benchmarks/cpuraytracer.ex 256 20 1

#mix run benchmarks/gpudp.ex 33792 1
#mix run benchmarks/cpudp.ex 33792 1



#nvcc -o c_implementations/craytracer c_implementations/craytracer.cu
#./c_implementations/craytracer