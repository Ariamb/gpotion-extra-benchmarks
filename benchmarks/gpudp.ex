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


defmodule FUNC do
  def fill_array(a, b, n, n) do 
    {a, b}
  end
  def fill_array(a, b, i, n) do
    fill_array(Matrex.set(a, 1, i + 1, i), Matrex.set(b, 1, i + 1, i), i+1, n)
  end  
end

{n, _} = Integer.parse(Enum.at(System.argv, 0))
{iteration, _} = Integer.parse(Enum.at(System.argv, 1))

list = [Enum.to_list(0..n-1)]

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

#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

resultreal = GPotion.get_gmatrex(ref3)
next = System.monotonic_time()
s = Matrex.sum(resultreal)

#next = System.monotonic_time()

text = "time: #{next - prev}, iteration: #{iteration}, array size: #{n} \n"
File.write!("time-GPUDP.txt", text, [:append])

#prev = System.monotonic_time()
#eresult = Matrex.add(vet1,vet2)
#next = System.monotonic_time()
#IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"