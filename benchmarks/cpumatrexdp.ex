import Matrex

{n, _} = Integer.parse(Enum.at(System.argv, 0))
{iteration, _} = Integer.parse(Enum.at(System.argv, 1))

a = Matrex.new([Enum.to_list(1..n)])
b = Matrex.new([Enum.to_list(1..n)])

IO.puts("Começando o cálculo de dot product")

prev = System.monotonic_time()
c = Matrex.dot_nt(a, b)
next = System.monotonic_time()

IO.puts("Cálculo finalizado. Resultado: #{c}")

text = "time: #{System.convert_time_unit(next - prev,:native,:microsecond)} \t iteration: #{iteration} \t array size: #{n} \n"
File.write!("time-matrex-cpudp.txt", text, [:append])


