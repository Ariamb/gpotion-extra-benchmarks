import Random
import Matrex
import Bitwise
#import GPotion

Random.seed(42)



defmodule RayTracer do
  #gpotion raytracing( s, ptr, dim,spheres, inf ) do
  gpotion raytracing(dim, spheres, image) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x
    
    ox = (x - dim/2)
    oy = (y - dim/2)

    r = 0.0
    g = 0.0
    b = 0.0

    maxz = -999999999
    for i in range(0, 20) do
      n = 0.0
      #{ #hit
      dx = ox - sphere[i * 7 + 4]
      dy = oy - sphere[i * 7 + 5]

      if (dx * dx + dy * dy) <  sphere[i * 7 + 3] * sphere[i * 7 + 3] do
        dzsqrd = sphere[i * 7 + 3] * sphere[i * 7 + 3] - dx * dx - dy * dy
        dz = sqrt(dzsqrd)
        n = dz / sqrt(sphere[i * 7 + 3] * sphere[i * 7 + 3])
        dz = dz + sphere[i * 7 + 6] 
      end
      #}

      if t > maxz do
        fscale = n
        r = sphere[i * 7 + 0] * fscale
        g = sphere[i * 7 + 1] * fscale
        b = sphere[i * 7 + 2] * fscale
        maxz = t
      end
    end

    image[offset * 4 + 0] = r * 255
    image[offset * 4 + 1] = g * 255
    image[offset * 4 + 2] = b * 255
    image[offset * 4 + 3] = 255
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
    
    def sphereMaker(spheres, max, max) do
      max = max - 1
        Matrex.set(spheres, 1, max * 7 + 1, Main.rnd(1)) 
        |> Matrex.set( 1, max * 7 + 2, Main.rnd(1)) #g
        |> Matrex.set( 1, max * 7 + 3, Main.rnd(1)) #b
        |> Matrex.set( 1, max * 7 + 4, Main.rnd(100) + 20) #radius
        |> Matrex.set( 1, max * 7 + 5, Main.rnd(1000) - 500) #x
        |> Matrex.set( 1, max * 7 + 6, Main.rnd(1000) - 500) #y
        |> Matrex.set( 1, max * 7 + 7, Main.rnd(1000) - 500) #z
    end
    def sphereMaker(spheres, n, max) do
      n = n - 1
      spheres  = Matrex.set(spheres, 1, n * 7 + 1, Main.rnd(1)) 
      |> Matrex.set( 1, n * 7 + 2, Main.rnd(1)) #g
      |> Matrex.set( 1, n * 7 + 3, Main.rnd(1)) #b
      |> Matrex.set( 1, n * 7 + 4, Main.rnd(100) + 20) #radius
      |> Matrex.set( 1, n * 7 + 5, Main.rnd(1000) - 500) #x
      |> Matrex.set( 1, n * 7 + 6, Main.rnd(1000) - 500) #y
      |> Matrex.set( 1, n * 7 + 7, Main.rnd(1000) - 500) #z
      |> sphereMaker(n + 2, max)
    end

    def all do
        
    end

    def main do
        sphereList = Matrex.zeros(1, 20 * 7)
        sphereList = sphereMaker(sphereList, 1, 20)
        IO.inspect(sphereList)
        kernel = GPotion.load(&RayTracer.raytracing)
        
        refSphere = GPotion.new_gmatrex(sphereList)
        refImag = GPotion.new_gmatrex(1, 1024 * 1024 * 4)
        #GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[1024, refSphere, refImag])
        
        GPotion.spawn(kernel,{64,64,1},{8,8,1},[1024, refSphere, refImag])
        GPotion.synchronize()
        

        #color = CPUraytracer.loopSpheres(sphereList, {0, 0, 0}, {1, 1}, 0, 20, CPUraytracer.minusinf)
        #sphereLocal = Enum.at(sphereList, 19)
        #image = Matrex.zeros(1, (CPUraytracer.dim + 1)* (CPUraytracer.dim + 1) * 4)
        #image = CPUraytracer.kernelLoop(sphereList, image, 1, 1)
        #IO.inspect(image)


        #width = 1024
        #height = 1024

        #widthInBytes = width * Bmpgen.bytes_per_pixel

        #paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        #stride = widthInBytes + paddingSize

        #IO.puts("ray tracer completo, começando escrita")
        #Bmpgen.writeFileHeader(height, stride)
        #Bmpgen.writeInfoHeader(height, width)
        #Bmpgen.recursiveWrite(image, 1, (CPUraytracer.dim + 1)* (CPUraytracer.dim + 1) * 4)

        

    end
end

Main.main
  


