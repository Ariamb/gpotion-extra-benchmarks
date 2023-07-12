defmodule DP do
  import GPotion

gpotion add_vectors(ref4,ref3, a, b, n, tpb) do
  __shared__ cache[tpb]

  tid = threadIdx.x + blockIdx.x * blockDim.x;
  stride = blockDim.x * gridDim.x;

  #cacheIndex = threadIdx.x
  #temp = 0.0
  
  #while (tid < n) do
  #  temp = a[tid] * b[tid] + temp
  #  tid = blockDim.x * gridDim.x + tid
  #end
      
  #cache[cacheIndex] = temp
  #__syncthreads()
      
  #i = blockDim.x/2
  #while (i != 0) do
  #  if (cacheIndex < i) do
  #    cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex]
  #  end
  #  __syncthreads()
  #  i = i/2
  #end
  
  #if (cacheIndex == 0) do
  #  ref4[blockIdx.x] = cache[0]
  #end
  ref3[0] = ref3[0]
  tpb = tpb+ 0
  index = threadIdx.x + blockIdx.x * blockDim.x;
  for j in range(index,n,stride) do
    ref3[j] = a[j] * b[j]
  end
  __syncthreads()
end
end

n = 10000000



list = [Enum.to_list(1..n)]

vet1 = Matrex.new(list)
vet2 = Matrex.new(list)
vet3 = Matrex.ones(1,n)
vet4 = Matrex.ones(1,n)



kernel=GPotion.load(&DP.add_vectors/5)

threadsPerBlock = 128;
numberOfBlocks = div(n + threadsPerBlock - 1, threadsPerBlock)


#prev = System.monotonic_time()

ref1=GPotion.new_gmatrex(vet1)
ref2=GPotion.new_gmatrex(vet2)
ref3=GPotion.new_gmatrex(vet3)
ref4=GPotion.new_gmatrex(vet4)

tpb = 256
GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref4,ref3, ref1,ref2,n, tpb])
GPotion.synchronize()

#next = System.monotonic_time()
#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

resultreal = GPotion.get_gmatrex(ref3)
resultfake = GPotion.get_gmatrex(ref4)
IO.puts("real")
IO.inspect resultreal
IO.puts("fake")
IO.inspect resultfake


#prev = System.monotonic_time()
#eresult = Matrex.add(vet1,vet2)
#next = System.monotonic_time()
#IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

#diff = Matrex.subtract(result,eresult)

#IO.puts "this value must be zero: #{Matrex.sum(diff)}"