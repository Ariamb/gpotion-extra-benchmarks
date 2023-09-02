import Random
import Matrex
import Bitwise

Random.seed(313)
defmodule CPUraytracer do
    def kernel(spheres, image, {x, y}) do

        ox = x - CPUraytracer.dim / 2
        oy = y - CPUraytracer.dim / 2
        #offset = (x + y * CPUraytracer.dim) * 4 + 1

        {r, g, b} = loopSpheres(spheres, {0, 0, 0}, {ox, oy}, 0, length(spheres), CPUraytracer.minusinf)
        #image = Matrex.set(image, 1, offset + 0, b)
        #image = Matrex.set(image, 1, offset + 1, g)
        #image = Matrex.set(image, 1, offset + 2, r)
        #image = Matrex.set(image, 1, offset + 3, 255)
        [b, g, r, 255 | image]
    end

    def kernelLoop(spheres, image, 257, 257) do #257, 257
        image
    end
    
    #def kernelLoop(spheres, image, i, 257) do #257
    #    #IO.puts("#{i}/#{CPUraytracer.dim}")
    #    CPUraytracer.kernelLoop(spheres, image, i + 1, 1)
    #end
    
    def kernelLoop(spheres, image, 257, j) do #257
        #IO.puts("#{i}/#{CPUraytracer.dim}")
        CPUraytracer.kernelLoop(spheres, image, 1, j+1)
    end

    def kernelLoop(spheres, image, i, j) do
        CPUraytracer.kernel(spheres, CPUraytracer.kernelLoop(spheres, image, i+1, j),{i, j})
    end

    def loopSpheres(sphereList, color, {x, y}, maxi, maxi, maxz) do
        if y >= 128 or x >= 128 do
            #IO.puts("#{x}/256")    
        end
        color

        
    end


    def overflowFix(color) do #for values bigger than 255
        if color > 255 do
            color = 255
        else
            color
        end

    end
    def loopSpheres(sphereList, color, {ox, oy}, i, maxi, maxz) do
        sphereLocal = Enum.at(sphereList, i)
        {n, z} = Sphere.hit(sphereLocal, ox, oy)
        {r, g, b} = color
        if maxz != -999 and z != -999 do
            #IO.puts("maxz: #{maxz} candidatez: #{z}, current sphere number: #{i}")    
        end
        if  z > maxz do
            
            
            loopSpheres(sphereList, {
                overflowFix(sphereLocal.r * n * 255),
                overflowFix(sphereLocal.g * n * 255),
                overflowFix(sphereLocal.b * n * 255)
            }, {ox, oy}, i + 1, maxi, z)
        else
            loopSpheres(sphereList, color, {ox, oy}, i + 1, maxi, z)
        end
        

        #    maxz = z
        #end
        #loopSpheres(color, spheres, i + 1, maxi, pos, maxz)

    end
    
    def dim do
        256
        
    end
    def minusinf do
        -999
    end
end


defmodule Sphere do
    defstruct r: 0, g: 0, b: 0, radius: 0, x: 0, y: 0, z: 0

    def new() do
        %Sphere{}
    end
    
    def hit(sphere, ox, oy) do
        dx = ox - sphere.x
        dy = oy - sphere.y
        if (dx * dx + dy * dy) < sphere.radius * sphere.radius do
            dz = :math.sqrt(sphere.radius * sphere.radius - dx * dx - dy * dy)
            n = dz / :math.sqrt(sphere.radius * sphere.radius)
            return = {n, dz + sphere.z}
        else
            return = {0, CPUraytracer.minusinf} #makeshift infinity in elixir, using 32 memory bits
        end
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
  def recursiveWrite([b, g, r, 255 | image]) do
    l = [<<trunc(r)>>, <<trunc(b)>>, <<trunc(g)>>, <<255>>]
    File.write!("img-cpu-999.bmp", l, [:append])
    recursiveWrite(image)
    

  end

  def writeFileHeader(height, stride) do
    fileSize = Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize + (stride * height)    
    fileHeader = ['B'] ++ ['M'] ++ [<<fileSize>>] ++ [<<fileSize >>> 8>>] ++ [<<fileSize >>> 16>>] ++ [<<fileSize >>> 24>>] ++ List.duplicate(<<0>>, 4) ++ [<<Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3)
    #IO.inspect(fileHeader)
    IO.puts("\n-----------------------\n")
    File.write!("img-cpu-999.bmp", fileHeader)
  end
  def writeInfoHeader(height, width) do
    
    infoHeader = [<<Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3) ++ [<<width>>, <<width >>> 8>>, <<width >>> 16>>, <<width >>> 24>>, <<height>>, <<height >>> 8>>, <<height >>> 16>>, <<height >>> 24>>, <<1>>, <<0>>, <<Bmpgen.bytes_per_pixel * 8>>] ++ List.duplicate(<<0>>, 25)
    #IO.inspect(infoHeader)
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
            x: Main.rnd(256) - 128,
            y: Main.rnd(256) - 128,
            z: Main.rnd(256) - 128,
        }]
    end
    def sphereMaker(n) do
        [%Sphere{
            r: Main.rnd(1),
            g: Main.rnd(1),
            b: Main.rnd(1),
            radius: Main.rnd(20) + 5,
            x: Main.rnd(256) - 128,
            y: Main.rnd(256) - 128,
            z: Main.rnd(256) - 128,
        }] ++ sphereMaker(n - 1)
    end

    def all do
        
    end

    def main do
        sphereList = sphereMaker(20)
        IO.inspect(sphereList)
        #color = CPUraytracer.loopSpheres(sphereList, {0, 0, 0}, {1, 1}, 0, 20, CPUraytracer.minusinf)
        #sphereLocal = Enum.at(sphereList, 19)
        #image = Matrex.zeros(1, (CPUraytracer.dim + 1)* (CPUraytracer.dim + 1) * 4)
        image = []
        prev = System.monotonic_time()
        image = CPUraytracer.kernelLoop(sphereList, image, 1, 1)
        next = System.monotonic_time()
        IO.puts("tempo: #{next - prev}")
        IO.inspect(image)


        width = 256
        height = 256

        widthInBytes = width * Bmpgen.bytes_per_pixel

        paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        stride = widthInBytes + paddingSize

        IO.puts("ray tracer completo, começando escrita")
        Bmpgen.writeFileHeader(height, stride)
        Bmpgen.writeInfoHeader(height, width)
        #Bmpgen.recursiveWrite(image, 1, (CPUraytracer.dim + 1)* (CPUraytracer.dim + 1) * 4)
        Bmpgen.recursiveWrite(image)
        

    end
end

Main.main

