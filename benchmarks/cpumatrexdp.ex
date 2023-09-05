import Matrex


defmodule Utils do
    def dot_product([],[], sum) do
        sum
    end
    def dot_product([a | as],[b | bs], sum) do
        dot_product(as, bs, a * b + sum)
    end
end 


{n, _} = Integer.parse(Enum.at(System.argv, 0))
{iteration, _} = Integer.parse(Enum.at(System.argv, 1))


a = Matrex.new([Enum.to_list(1..n)])
b = Matrex.new([Enum.to_list(1..n)])

IO.puts("começando a calcular")
prev = System.monotonic_time()
c = Matrex.dot_nt(a, b)
next = System.monotonic_time()
IO.puts("terminei de calcular")
IO.inspect(c)

text = "time: #{System.convert_time_unit(next - prev,:native,:microsecond)} \t iteration: #{iteration} \t array size: #{n} \n"
File.write!("time-matrex-cpudp.txt", text, [:append])


