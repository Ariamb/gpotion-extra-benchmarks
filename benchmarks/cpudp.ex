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


#a = Matrex.new([Enum.to_list(1..n)])
#b = Matrex.new([Enum.to_list(1..n)])
a = Enum.to_list(1..n)
b = Enum.to_list(1..n)

#c = Matrex.dot_nt(a, b)
prev = System.monotonic_time()
c = Utils.dot_product(a, b, 0)
next = System.monotonic_time()

IO.inspect(c)

text = "time: #{System.convert_time_unit(next - prev,:native,:microsecond)} \t iteration: #{iteration} \t array size: #{n} \n"
File.write!("time-cpudp.txt", text, [:append])


