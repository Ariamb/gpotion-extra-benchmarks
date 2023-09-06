#!/bin/bash

N=30
SPHERES=20
# Compiling files
sh compile.sh


for SIZE in 256 1024 3072; do # 256 1024 3072
    echo $SIZE

    for i in $(seq 1 $N); do
        echo $i
    sleep 1

	echo C GPU
        ./c_implementations/craytracer $SIZE $SPHERES $i
    sleep 0.2
    
    echo Elixir CPU
        mix run benchmarks/cpuraytracer.ex $SIZE $SPHERES $i
    sleep 0.2


    echo Elixir GPU
        mix run benchmarks/gpuraytracer.ex $SIZE $SPHERES $i
    done
    sleep 0.2
done



#mix run benchmarks/gpuraytracer.ex 256 20 1
#mix run benchmarks/cpuraytracer.ex 256 20 1

#mix run benchmarks/gpudp.ex 33792 1
#mix run benchmarks/cpudp.ex 33792 1



#nvcc -o c_implementations/craytracer c_implementations/craytracer.cu
#./c_implementations/craytracer