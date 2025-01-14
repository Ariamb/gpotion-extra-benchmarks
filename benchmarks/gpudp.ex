defmodule GPUDP do
  import GPotion
  gpotion dot_product(ref4, a, b, n) do

  __shared__ cache[256]

  tid = threadIdx.x + blockIdx.x * blockDim.x;
  cacheIndex = threadIdx.x
  temp = 0.0
  
  while (tid < n) do
    temp = a[tid] * b[tid] + temp
    tid = blockDim.x * gridDim.x + tid
  end
      
  cache[cacheIndex] = temp
  __syncthreads()
      
  i = blockDim.x/2
  while (i != 0) do
    if (cacheIndex < i) do
      cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex]
    end
    __syncthreads()
    i = i/2
  end
  
  if (cacheIndex == 0) do
    ref4[blockIdx.x] = cache[0]
  end

end
end

{n, _} = Integer.parse(Enum.at(System.argv, 0))
{iteration, _} = Integer.parse(Enum.at(System.argv, 1))

list = [Enum.to_list(1..n)]

vet1 = Matrex.new(list)
vet2 = Matrex.new(list)


threadsPerBlock = 256
blocksPerGrid = div(n + threadsPerBlock - 1, threadsPerBlock)
numberOfBlocks = blocksPerGrid

vet3 = Matrex.ones(1,blocksPerGrid)

prev = System.monotonic_time()

kernel=GPotion.load(&DP.dot_product/5)

ref1=GPotion.new_gmatrex(vet1)
ref2=GPotion.new_gmatrex(vet2)
ref3=GPotion.new_gmatrex(vet3)

GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref3, ref1,ref2,n])
GPotion.synchronize()

resultreal = GPotion.get_gmatrex(ref3)
s = Matrex.sum(resultreal)
next = System.monotonic_time()

IO.inspect(s)

text = "time: #{System.convert_time_unit(next - prev,:native,:microsecond)} \t iteration: #{iteration} \t array size: #{n} \n"
File.write!("time-GPUDP.txt", text, [:append])