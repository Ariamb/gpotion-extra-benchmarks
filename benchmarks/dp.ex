defmodule DP do
  import GPotion

#gpotion add_vectors(ref4,ref3, a, b, n, tpb) do
  gpotion add_vectors(ref4, a, b, n) do

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
  def compare_array(_c, _d, n, n) do
    #IO.puts("altered: #{c[i]} -")
    #IO.puts("original: #{d[i]} \n")
  end
  def compare_array(c, d, i, n) do
    IO.puts("pos: #{i};  base: #{Matrex.at(d, 1, i+1)};  altered: #{Matrex.at(c, 1, i+1)}  \n")
    
    #IO.puts("original: #{d[i]} \n")
    compare_array(c, d, i+1, n)
  end
  def soma_array([], s) do
    0
  end
  def soma_array([h|t], s) do
    soma_array(t, s+h)
  end
  
end

n = 33 * 1024



list = [Enum.to_list(1..n)]

vet1 = Matrex.new(list)
vet2 = Matrex.new(list)
threadsPerBlock = 256

blocksPerGrid = (n+threadsPerBlock-1) / threadsPerBlock
vet3 = Matrex.ones(1,blocksPerGrid)



kernel=GPotion.load(&DP.add_vectors/5)

threadsPerBlock = 128;
numberOfBlocks = div(n + threadsPerBlock - 1, threadsPerBlock)


#prev = System.monotonic_time()

ref1=GPotion.new_gmatrex(vet1)
ref2=GPotion.new_gmatrex(vet2)
ref3=GPotion.new_gmatrex(vet3)


GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref3, ref1,ref2,n])
GPotion.synchronize()

#next = System.monotonic_time()
#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

#resultfake = GPotion.get_gmatrex(ref3)
resultreal = GPotion.get_gmatrex(ref3)
IO.puts("rel")
IO.inspect resultreal
#IO.puts("fake")
#IO.inspect resultfake


#prev = System.monotonic_time()
#eresult = Matrex.add(vet1,vet2)
#next = System.monotonic_time()
#IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

#diff = Matrex.subtract(result,eresult)

#IO.puts "this value must be zero: #{Matrex.sum(diff)}"