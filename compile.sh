make
mix deps.get
mix compile

nvcc -o c_implementations/craytracer c_implementations/craytracer.cu
nvcc -o c_implementations/dotproduct c_implementations/dotproduct.cu
