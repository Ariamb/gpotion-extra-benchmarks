import Bitwise

Random.seed(313)



defmodule RayTracer do
import GPotion
 
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
  def recursiveWrite(_image, max, max) do
    IO.puts("done!")
  end

  def recursiveWrite(image, i, max) do
    x = trunc(Matrex.at(image, 1, i))
    File.write!("img-gpuraytracer-#{Main.dim}.bmp", <<x>>, [:append])
    recursiveWrite(image, i+1, max)
  end

  def writeFileHeader(height, stride) do
    fileSize = Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize + (stride * height)    
    fileHeader = ['B'] ++ ['M'] ++ [<<fileSize>>] ++ [<<fileSize >>> 8>>] ++ [<<fileSize >>> 16>>] ++ [<<fileSize >>> 24>>] ++ List.duplicate(<<0>>, 4) ++ [<<Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3)
    IO.puts("\n-----------------------\n")
    File.write!("img-gpuraytracer-#{Main.dim}.bmp", fileHeader)
  end
  def writeInfoHeader(height, width) do
    
    infoHeader = [<<Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3) ++ [<<width>>, <<width >>> 8>>, <<width >>> 16>>, <<width >>> 24>>, <<height>>, <<height >>> 8>>, <<height >>> 16>>, <<height >>> 24>>, <<1>>, <<0>>, <<Bmpgen.bytes_per_pixel * 8>>] ++ List.duplicate(<<0>>, 25)
    #IO.inspect(infoHeader)
    File.write!("img-gpuraytracer-#{Main.dim}.bmp", infoHeader, [:append])
  end
end


defmodule Main do
    def rnd(x) do
        x * Random.randint(1, 32767) / 32767
    end
    
    def sphereMaker(spheres, max, max) do
      max = max - 1
      #Main.rnd(1)
      #Main.rnd(1)
      #Main.rnd(1)
        Matrex.set(spheres, 1, max * 7 + 1, Main.rnd(1)) 
        |> Matrex.set( 1, max * 7 + 2, Main.rnd(1)) #g
        |> Matrex.set( 1, max * 7 + 3, Main.rnd(1)) #b
        |> Matrex.set( 1, max * 7 + 4, Main.rnd(20) + 5) #radius
        |> Matrex.set( 1, max * 7 + 5, Main.rnd(256) - 128) #x
        |> Matrex.set( 1, max * 7 + 6, Main.rnd(256) - 128) #y
        |> Matrex.set( 1, max * 7 + 7, Main.rnd(256) - 128) #z
    end
    def sphereMaker(spheres, n, max) do
      
      Matrex.set(spheres, 1, n * 7 + 1, Main.rnd(1)) 
      |> Matrex.set( 1, (n - 1) * 7 + 2, Main.rnd(1)) #g
      |> Matrex.set( 1, (n - 1) * 7 + 3, Main.rnd(1)) #b
      |> Matrex.set( 1, (n - 1) * 7 + 4, Main.rnd(20) + 5) #radius
      |> Matrex.set( 1, (n - 1) * 7 + 5, Main.rnd(256) - 128) #x
      |> Matrex.set( 1, (n - 1) * 7 + 6, Main.rnd(256) - 128) #y
      |> Matrex.set( 1, (n - 1) * 7 + 7, Main.rnd(256) - 128) #z
      |> sphereMaker(n + 1, max)
    end

    def dim do
      {d, _} = Integer.parse(Enum.at(System.argv, 0))
      d
    end
    def spheres do
      {s, _} = Integer.parse(Enum.at(System.argv, 1))
      s
    end

    def main do
        sphereList = Matrex.zeros(1, Main.spheres * 7)
        sphereList = sphereMaker(sphereList, 1, Main.spheres)

        IO.inspect(sphereList)

        
        width = Main.dim
        height = Main.dim

        imageList = Matrex.zeros(1, (width + 1) * (height + 1) * 4)

        prev = System.monotonic_time()
        kernel = GPotion.load(&RayTracer.raytracing/3)
        
        refSphere = GPotion.new_gmatrex(sphereList)
        refImag = GPotion.new_gmatrex(imageList)

        GPotion.spawn(kernel,{trunc(width/16),trunc(height/16),1},{16,16,1},[width, refSphere, refImag])
        GPotion.synchronize()
        
        image = GPotion.get_gmatrex(refImag)
        next = System.monotonic_time()

        widthInBytes = width * Bmpgen.bytes_per_pixel

        paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        stride = widthInBytes + paddingSize

        IO.puts("ray tracer completo, começando escrita")
        Bmpgen.writeFileHeader(height, stride)
        Bmpgen.writeInfoHeader(height, width)
        Bmpgen.recursiveWrite(image, 1, (width+1)* (height+1) * 4)

        {iteration, _} = Integer.parse(Enum.at(System.argv, 2))
        text = "time: #{next - prev}, iteration: #{iteration}, dimension: #{height}x#{width}, spheres: #{Main.spheres} \n"

        File.write!("time-gpuraytracer.txt", text, [:append])
      
    end
end

Main.main
  


