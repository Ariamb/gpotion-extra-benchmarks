import Random
import Matrex
import Bitwise

Random.seed(313)



defmodule RayTracer do
import GPotion

  #gpotion raytracing( s, ptr, dim,spheres, inf ) do
 
gpotion raytracing(dim, spheres, image) do

  x = threadIdx.x + blockIdx.x * blockDim.x
  y = threadIdx.y + blockIdx.y * blockDim.y
  offset = x + y * blockDim.x * gridDim.x
  #testando se poderiam ser os comments
  ox = (x - dim/2)
  oy = (y - dim/2)

  r = 0.0
  g = 0.0
  b = 0.0

  maxz = -99999.0

  for i in range(0, 20) do
    
    sphereR = spheres[i * 7 + 0]
    sphereG = spheres[i * 7 + 1]
    sphereB = spheres[i * 7 + 2]
    sphereRadius = spheres[i * 7 + 3]
    sphereX = spheres[i * 7 + 4]
    sphereY = spheres[i * 7 + 5]
    sphereZ = spheres[i * 7 + 6]
  
    dx = ox - sphereX
    dy = oy - sphereY
    n = 0.0
    t = 0.0
    dz = 0.0
    if (dx * dx + dy * dy) <  (sphereRadius * sphereRadius) do
      dz = sqrtf(sphereRadius * sphereRadius - (dx * dx) - (dy * dy))
      n = dz / sqrtf(sphereRadius * sphereRadius)
      t = dz + sphereZ
    else 
      t = -99999.0
      n = 0.0
    end

    if t > maxz do
      fscale = n
      r = sphereR * fscale
      g = sphereG * fscale
      b = sphereB * fscale
      maxz = t
    end
    
    
  end
  #image[0] = 1 + 0 * (dim + sphere[i * 7 ] + 0 * (r + g + b + oy + ox + maxz + n + dx + dy + dz + fscale) )
  image[offset * 4 + 0] = r * 255 
  image[offset * 4 + 1] = b * 255
  image[offset * 4 + 2] = g * 255
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
    File.write!("imgAGORAVAI.bmp", <<x>>, [:append])
    recursiveWrite(image, i+1, max)
  end

  def writeFileHeader(height, stride) do
    fileSize = Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize + (stride * height)    
    fileHeader = ['B'] ++ ['M'] ++ [<<fileSize>>] ++ [<<fileSize >>> 8>>] ++ [<<fileSize >>> 16>>] ++ [<<fileSize >>> 24>>] ++ List.duplicate(<<0>>, 4) ++ [<<Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3)
    #IO.inspect(fileHeader)
    IO.puts("\n-----------------------\n")
    File.write!("imgAGORAVAI.bmp", fileHeader)
  end
  def writeInfoHeader(height, width) do
    
    infoHeader = [<<Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3) ++ [<<width>>, <<width >>> 8>>, <<width >>> 16>>, <<width >>> 24>>, <<height>>, <<height >>> 8>>, <<height >>> 16>>, <<height >>> 24>>, <<1>>, <<0>>, <<Bmpgen.bytes_per_pixel * 8>>] ++ List.duplicate(<<0>>, 25)
    #IO.inspect(infoHeader)
    File.write!("imgAGORAVAI.bmp", infoHeader, [:append])
  end
end


defmodule Main do
    def rnd(x) do
        x * Random.randint(1, 32767) / 32767
    end
    
    def sphereMaker(spheres, max, max) do
      max = max - 1
      Main.rnd(1)
      Main.rnd(1)
      Main.rnd(1)
        Matrex.set(spheres, 1, max * 7 + 1, 1) 
        |> Matrex.set( 1, max * 7 + 2, 0) #g
        |> Matrex.set( 1, max * 7 + 3, 0) #b
        |> Matrex.set( 1, max * 7 + 4, Main.rnd(20) + 5) #radius
        |> Matrex.set( 1, max * 7 + 5, Main.rnd(256) - 128) #x
        |> Matrex.set( 1, max * 7 + 6, Main.rnd(256) - 128) #y
        |> Matrex.set( 1, max * 7 + 7, Main.rnd(256) - 128) #z
    end
    def sphereMaker(spheres, n, max) do
      #n = n - 1
      spheres  = Matrex.set(spheres, 1, n * 7 + 1, Main.rnd(1)) 
      |> Matrex.set( 1, (n - 1) * 7 + 2, Main.rnd(1)) #g
      |> Matrex.set( 1, (n - 1) * 7 + 3, Main.rnd(1)) #b
      |> Matrex.set( 1, (n - 1) * 7 + 4, Main.rnd(20) + 5) #radius
      |> Matrex.set( 1, (n - 1) * 7 + 5, Main.rnd(256) - 128) #x
      |> Matrex.set( 1, (n - 1) * 7 + 6, Main.rnd(256) - 128) #y
      |> Matrex.set( 1, (n - 1) * 7 + 7, Main.rnd(256) - 128) #z
      |> sphereMaker(n + 1, max)
    end

    def all do
        
    end

    def main do
        sphereList = Matrex.zeros(1, 20 * 7)
        sphereList = sphereMaker(sphereList, 1, 20)

        IO.inspect(sphereList)
        kernel = GPotion.load(&RayTracer.raytracing/3)
        
        refSphere = GPotion.new_gmatrex(sphereList)
        imageList = Matrex.zeros(1, (256+1) * (256+1) * 4)
        IO.inspect(imageList[:size])

        refImag = GPotion.new_gmatrex(imageList)
        IO.inspect(imageList)

        #GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[256, refSphere, refImag])

        GPotion.spawn(kernel,{trunc(256/16),trunc(256/16),1},{16,16,1},[256, refSphere, refImag])
        GPotion.synchronize()
        
        image = GPotion.get_gmatrex(refImag)
        IO.inspect(image)
        {a, b} =  image[:size]
        IO.puts(Matrex.at(image, 1, trunc(b/2)))
        width = 256
        height = 256

        widthInBytes = width * Bmpgen.bytes_per_pixel

        paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        stride = widthInBytes + paddingSize

        IO.puts("ray tracer completo, começando escrita")
        IO.inspect(image[:size])
        Bmpgen.writeFileHeader(height, stride)
        Bmpgen.writeInfoHeader(height, width)
        Bmpgen.recursiveWrite(image, 1, (256+1)* (256+1) * 4)

        
        

        

    end
end

Main.main
  


