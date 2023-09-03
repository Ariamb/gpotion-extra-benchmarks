import Matrex

defmodule Utils do
    import Matrex

    def fill_array(a, b, n, n) do 
        {a, b}
    end


    def fill_array(a, b, i, n) do
        fill_array(Matrex.set(a, 1, i + 1, i), Matrex.set(b, 1, i + 1, i), i+1, n)
    end
    
  end
  

#n = 33 * 1024
{n, _} = Integer.parse(Enum.at(System.argv, 0))
{iteration, _} = Integer.parse(Enum.at(System.argv, 1))


a = Matrex.ones(1, n)
b = Matrex.ones(1, n)

{a, b} = Utils.fill_array(a, b, 0, n)

prev = System.monotonic_time()
c = Matrex.dot_nt(a, b)
next = System.monotonic_time()

IO.inspect(c)

text = "time: #{next - prev}, iteration: #{iteration}, array size: #{n} \n"
File.write!("time-cpudp.txt", text, [:append])


