defmodule Kernel do
    import GPotion

    gpotion add_vectors(result, a, b, n) do
        index = threadIdx.x + blockIdx.x * blockDim.x

        if (index < n) do
            result[index] = a[index] + b[index]
        end
    end
end


n = 33 * 1024

list = [Enum.to_list(1..n)]

cpu_a = Matrex.new(list)
cpu_b = Matrex.new(list)


kernel = GPotion.load(&Add_vectors.add/4)

threadsPerBlock = 256
numberOfBlocks = div(n + threadsPerBlock - 1, threadsPerBlock)

gpu_a = GPotion.new_gmatrex(cpu_a)
gpu_b = GPotion.new_gmatrex(cpu_b)

result = GPotion.new_gmatrex(1, n)

GPotion.spawn(kernel, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [result, gpu_a, gpu_b, n])
GPotion.synchronize()

result = GPotion.get_gmatrex(result)
IO.inspect(result)
