make
mix deps.get
mix compile
 
rm c_implementations/cdotproduct
rm c_implementations/craytracer
nvcc -o c_implementations/craytracer c_implementations/craytracer.cu
nvcc -o c_implementations/cdotproduct c_implementations/cdotproduct.cu
