
import Bitwise

Random.seed(313)
defmodule CPUraytracer do
    def kernel(spheres, image, {x, y}, dim) do

        ox = x - dim / 2
        oy = y - dim / 2
        {r, g, b} = loopSpheres(spheres, {0, 0, 0}, {ox, oy}, CPUraytracer.minusinf)
        [r, g, b, 255 | image]
    end

    def kernelLoop(_, image, max, max, max) do 
        image
    end
    
    
    
    def kernelLoop(spheres, image, max, j, max) do 
        CPUraytracer.kernelLoop(spheres, image, 0, j+1, max)
    end

    def kernelLoop(spheres, image, i, j, max) do
        CPUraytracer.kernel(spheres, CPUraytracer.kernelLoop(spheres, image, i+1, j, max),{i, j}, max)
    end

    def loopSpheres([], color, _,  _) do
        #if x >= CPUraytracer.dim / 2 do
        #    IO.puts("#{y + CPUraytracer/2 - 1}/CPUraytracer.dim")    
        #end
        color
    end

    def loopSpheres([sphereLocal | sphereList], color, {ox, oy}, maxz) do
        dx = ox - sphereLocal.x
        dy = oy - sphereLocal.y
        {n, t} = if (dx * dx + dy * dy) < sphereLocal.radius * sphereLocal.radius do
            dz = :math.sqrt(sphereLocal.radius * sphereLocal.radius - dx * dx - dy * dy)
            n = dz / :math.sqrt(sphereLocal.radius * sphereLocal.radius)
            {n, dz + sphereLocal.z}
        else
            {0, CPUraytracer.minusinf}
        end
        

        if  t > maxz do
            maxz = t
            loopSpheres(sphereList, {
                sphereLocal.r * n * 255,
                sphereLocal.g * n * 255,
                sphereLocal.b * n * 255
            }, {ox, oy}, maxz)
        else
            loopSpheres(sphereList, color, {ox, oy}, maxz)
        end
    end
    
    def dim do
        {d, _} = Integer.parse(Enum.at(System.argv, 0))
        d
    end
    def spheres do
        {s, _} = Integer.parse(Enum.at(System.argv, 1))
        s
    end
    def minusinf do
        -999999
    end
end


defmodule Sphere do
    defstruct r: 0, g: 0, b: 0, radius: 0, x: 0, y: 0, z: 0

    def new() do
        %Sphere{}
    end
end


defmodule Bmpgen do

  def fileHeaderSize do #constant
    14
  end

  def infoHeaderSize do #constant
    40
  end

  def bytes_per_pixel do
    4
  end
  #def recursiveWrite(something, max, max) do
  def recursiveWrite([]) do
    IO.puts("done opening!")
  end

  #def recursiveWrite([a | image], i, max) do
  def recursiveWrite([r, g, b, 255 | image]) do
    l = [<<trunc(r)>>, <<trunc(b)>>, <<trunc(g)>>, <<255>>]
    File.write!("img-cpu-999.bmp", l, [:append])
    recursiveWrite(image)
    

  end

  def writeFileHeader(height, stride) do
    fileSize = Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize + (stride * height)    
    fileHeader = ['B'] ++ ['M'] ++ [<<fileSize>>] ++ [<<fileSize >>> 8>>] ++ [<<fileSize >>> 16>>] ++ [<<fileSize >>> 24>>] ++ List.duplicate(<<0>>, 4) ++ [<<Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3)
    IO.puts("\n-----------------------\n")
    File.write!("img-cpu-999.bmp", fileHeader)
  end
  def writeInfoHeader(height, width) do
    
    infoHeader = [<<Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3) ++ [<<width>>, <<width >>> 8>>, <<width >>> 16>>, <<width >>> 24>>, <<height>>, <<height >>> 8>>, <<height >>> 16>>, <<height >>> 24>>, <<1>>, <<0>>, <<Bmpgen.bytes_per_pixel * 8>>] ++ List.duplicate(<<0>>, 25)
    File.write!("img-cpu-999.bmp", infoHeader, [:append])
  end
end

defmodule Main do
    def rnd(x) do
        x * Random.randint(1, 32767) / 32767
    end
    
    def sphereMaker(1) do
        [%Sphere{
            r: Main.rnd(1),
            g: Main.rnd(1),
            b: Main.rnd(1),
            radius: Main.rnd(20) + 5,
            x: Main.rnd(CPUraytracer.dim) - trunc(CPUraytracer.dim / 2),
            y: Main.rnd(CPUraytracer.dim) - trunc(CPUraytracer.dim / 2),
            z: Main.rnd(CPUraytracer.dim) - trunc(CPUraytracer.dim / 2),
        }]
    end
    def sphereMaker(n) do
        [%Sphere{
            r: Main.rnd(1),
            g: Main.rnd(1),
            b: Main.rnd(1),
            radius: Main.rnd(20) + 5,
            x: Main.rnd(CPUraytracer.dim) - trunc(CPUraytracer.dim / 2),
            y: Main.rnd(CPUraytracer.dim) - trunc(CPUraytracer.dim / 2),
            z: Main.rnd(CPUraytracer.dim) - trunc(CPUraytracer.dim / 2),
        } | sphereMaker(n - 1)]
    end

    def all do
        
    end

    def main do
        sphereList = sphereMaker(CPUraytracer.spheres)

        
        width = CPUraytracer.dim
        height = width #square image

        prev = System.monotonic_time()
        image = CPUraytracer.kernelLoop(sphereList, [], 0, 0, width)
        next = System.monotonic_time()
        IO.puts("tempo: #{next - prev}")
        IO.inspect(image)


        widthInBytes = width * Bmpgen.bytes_per_pixel

        paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        stride = widthInBytes + paddingSize

        IO.puts("ray tracer completo, começando escrita")
        Bmpgen.writeFileHeader(height, stride)
        Bmpgen.writeInfoHeader(height, width)
        Bmpgen.recursiveWrite(image)
        
        {iteration, _} = Integer.parse(Enum.at(System.argv, 2))
        text = "time: #{next - prev}, iteration: #{iteration}, dimension: #{height}x#{width}, spheres: #{CPUraytracer.spheres} \n"

        File.write!("time-cpuraytracer.txt", text, [:append])
    end
end

Main.main

