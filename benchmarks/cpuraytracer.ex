import Random
import Matrex
import Bitwise

Random.seed(42)
defmodule CPUraytracer do
    def kernel(spheres, image, {x, y}) do

        ox = x - CPUraytracer.dim / 2
        oy = y - CPUraytracer.dim / 2

        {r, g, b} = loopSpheres(spheres, {0, 0, 0}, {x, y}, 0, length(spheres), CPUraytracer.minusinf)
        offset = (x + y * CPUraytracer.dim) * 4 + 1
        image = Matrex.set(image, 1, offset + 0, b)
        image = Matrex.set(image, 1, offset + 1, g)
        image = Matrex.set(image, 1, offset + 2, r)
        image = Matrex.set(image, 1, offset + 3, 255)
        image
    end

    def kernelLoop(spheres, image, 1025, 1025) do #1025, 1025
        image
    end
    
    def kernelLoop(spheres, image, i, 1025) do #1025
        #IO.puts("#{i}/#{CPUraytracer.dim}")
        CPUraytracer.kernelLoop(spheres, image, i + 1, 1)
    end

    def kernelLoop(spheres, image, i, j) do
        CPUraytracer.kernel(spheres, CPUraytracer.kernelLoop(spheres, image, i, j+1),{i, j})
    end

    def loopSpheres(sphereList, color, {x, y}, maxi, maxi, maxz) do
        if y >= 1024 do
            IO.puts("#{x}/1024")    
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
        if r > 255 or g > 255 or b > 255 do
            IO.puts("COLOR OVERFLOW: #{r} #{g} #{b}")
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
        1024
        
    end
    def minusinf do
        -999999999
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
  def recursiveWrite(image, max, max) do
    IO.puts("done!")
  end

  def recursiveWrite(image, i, max) do
    x = trunc(Matrex.at(image, 1, i))
    File.write!("img.bmp", <<x>>, [:append])
    recursiveWrite(image, i+1, max)
  end

  def writeFileHeader(height, stride) do
    fileSize = Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize + (stride * height)    
    fileHeader = ['B'] ++ ['M'] ++ [<<fileSize>>] ++ [<<fileSize >>> 8>>] ++ [<<fileSize >>> 16>>] ++ [<<fileSize >>> 24>>] ++ List.duplicate(<<0>>, 4) ++ [<<Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3)
    #IO.inspect(fileHeader)
    IO.puts("\n-----------------------\n")
    File.write!("img.bmp", fileHeader)
  end
  def writeInfoHeader(height, width) do
    
    infoHeader = [<<Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3) ++ [<<width>>, <<width >>> 8>>, <<width >>> 16>>, <<width >>> 24>>, <<height>>, <<height >>> 8>>, <<height >>> 16>>, <<height >>> 24>>, <<1>>, <<0>>, <<Bmpgen.bytes_per_pixel * 8>>] ++ List.duplicate(<<0>>, 25)
    #IO.inspect(infoHeader)
    File.write!("img.bmp", infoHeader, [:append])
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
            radius: Main.rnd(100) + 20,
            x: Main.rnd(1000) - 500,
            y: Main.rnd(1000) - 500,
            z: Main.rnd(1000) - 500,
        }]
    end
    def sphereMaker(n) do
        [%Sphere{
            r: Main.rnd(1),
            g: Main.rnd(1),
            b: Main.rnd(1),
            radius: Main.rnd(100) + 20,
            x: Main.rnd(1000) - 500,
            y: Main.rnd(1000) - 500,
            z: Main.rnd(1000) - 500,
        }] ++ sphereMaker(n - 1)
    end

    def all do
        
    end

    def main do
        sphereList = sphereMaker(20)
        
        #color = CPUraytracer.loopSpheres(sphereList, {0, 0, 0}, {1, 1}, 0, 20, CPUraytracer.minusinf)
        #sphereLocal = Enum.at(sphereList, 19)
        image = Matrex.zeros(1, (CPUraytracer.dim + 1)* (CPUraytracer.dim + 1) * 4)
        image = CPUraytracer.kernelLoop(sphereList, image, 1, 1)
        IO.inspect(image)


        width = 1024
        height = 1024

        widthInBytes = width * Bmpgen.bytes_per_pixel

        paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        stride = widthInBytes + paddingSize

        IO.puts("ray tracer completo, começando escrita")
        Bmpgen.writeFileHeader(height, stride)
        Bmpgen.writeInfoHeader(height, width)
        Bmpgen.recursiveWrite(image, 1, (CPUraytracer.dim + 1)* (CPUraytracer.dim + 1) * 4)

        

    end
end

Main.main

