defmodule DP do
  import GPotion

  gpotion dotproduct(a,b,c) do

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cacheIndex = threadIdx.x
    temp = 0.0

    while (tid < 33 * 1024) do
      temp = a[tid] * b[tid] + temp
      tid = blockDim.x * gridDim.x + tid
    end
    cache[cacheIndex] = temp
    __syncthreads()

    while (i != 0) do
      if (cacheIndex < i) do
        cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex]
      __syncthreads()
      end
      i = i/2
    end

  end

  def fill_array(a, b, n, n) do 
    {a, b}
end
def fill_array(a, b, i, n) do
    fill_array(Matrex.set(a, 1, i + 1, i), Matrex.set(b, 1, i + 1, i), i+1, n)
end

end



n = 33 * 1024 #constante de tamanho q to usando


#só pra encher vetor


a = Matrex.ones(1, n)
b = Matrex.ones(1, n)
c = Matrex.ones(1, n)

{a, b} = DP.fill_array(a, b, 0, n)


threadsPerBlock = 256;
numberOfBlocks = trunc((n + threadsPerBlock - 1)/threadsPerBlock)


prev = System.monotonic_time()

kernel = GPotion.load(&DP.dotproduct/3)
a1 = GPotion.new_gmatrex(a)
b1 = GPotion.new_gmatrex(b)
c1 = GPotion.new_gmatrex(c)

GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[a1,b1,c1])

GPotion.synchronize()

_result = GPotion.get_gmatrex(c1)

next = System.monotonic_time()
#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
IO.puts "GPotion\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

IO.inspect result
#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])
