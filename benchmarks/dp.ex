defmodule DP do
  import GPotion

  gpotion dotproduct(a, b, c, tpb, n) do
    __shared__ cache[tpb]
    tid = threadIdx.x + blockIdx.x * blockDim.x
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
		  c[blockIdx.x] = cache[0]
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
  def compare_array(c, d, n, n) do
    #IO.puts("altered: #{c[i]} -")
    #IO.puts("original: #{d[i]} \n")
  end
  def compare_array(c, d, i, n) do
    IO.puts("altered: #{c[i]} - ")
    IO.puts("original: #{d[i]} \n")
    compare_array(c, d, i+1, n)
  end
end
  



n = 10 #constante de tamanho q to usando


#só pra encher vetor


a = Matrex.ones(1, n)
b = Matrex.ones(1, n)

c = GPotion.ones(1,n)
d = GPotion.ones(1,n)

c = GPotion.new_gmatrex(c)
d = GPotion.new_gmatrex(d)

IO.inspect c


{a, b} = FUNC.fill_array(a, b, 0, n)


threadsPerBlock = 256;
numberOfBlocks = trunc((n + threadsPerBlock - 1)/threadsPerBlock)


#prev = System.monotonic_time()

kernel = GPotion.load(&DP.dotproduct/3)
a1 = GPotion.new_gmatrex(a)
b1 = GPotion.new_gmatrex(b)

GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[a1,b1,c,threadsPerBlock,n])

GPotion.synchronize()

result = GPotion.get_gmatrex(c)

#FUNC.compare_array(result, d, 0, n)

IO.inspect result

#next = System.monotonic_time()
#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
#IO.puts "GPotion\t\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "


#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])
