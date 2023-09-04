import Bitwise

Random.seed(313)



defmodule RayTracer do
import GPotion
 
gpotion raytracing(dim, spheres, image) do

  x = threadIdx.x + blockIdx.x * blockDim.x
  y = threadIdx.y + blockIdx.y * blockDim.y
  offset = x + y * blockDim.x * gridDim.x
  #testando se poderiam ser os comments
  ox = 0.0
  oy = 0.0
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
    t = -99999.0
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
      r = spheres[i * 7 + 0] * fscale
      g = spheres[i * 7 + 1] * fscale
      b = spheres[i * 7 + 2] * fscale
      maxz = t
    end
    
    
  end
  image[offset * 4 + 3] = 255
  image[offset * 4 + 0] = r * 255 
  image[offset * 4 + 1] = g * 255
  image[offset * 4 + 2] = b * 255
  
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
  def recursiveWrite([]) do
    IO.puts("done opening!")
  end

  #def recursiveWrite([a | image], i, max) do
  def recursiveWrite([r, g, b, 255.0 | image]) do
    l = [<<trunc(g)>>, <<trunc(b)>>, <<trunc(r)>>, <<255>>]
    File.write!("img-gpuraytracer-#{Main.dim}.bmp", l, [:append])
    recursiveWrite(image)


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
    

    def sphereMaker2(1) do

          x = [
          Main.rnd(1),
          Main.rnd(1),
          Main.rnd(1),
          Main.rnd(20) + 5,
          Main.rnd(Main.dim) - trunc(Main.dim / 2),
          Main.rnd(Main.dim) - trunc(Main.dim / 2),
          Main.rnd(Main.dim) - trunc(Main.dim / 2)]
          IO.puts("sphere at 1 \n")
          IO.inspect(x)
          x
    end
    def sphereMaker2(n) do
      x = [
          Main.rnd(1),
          Main.rnd(1),
          Main.rnd(1),
          Main.rnd(20) + 5,
          Main.rnd(Main.dim) - trunc(Main.dim / 2),
          Main.rnd(Main.dim) - trunc(Main.dim / 2),
          Main.rnd(Main.dim) - trunc(Main.dim / 2)
      | sphereMaker2(n - 1)]
      IO.puts("sphere at #{n} \n")
      IO.inspect(x)
    end

    def spherePrinter([]) do
      File.write!("spheregpu.txt", "done\n", [:append])
      
    end
    def spherePrinter([ r, g, b, _radius, _x, _y, _z | list]) do
      File.write!("spheregpu.txt", "\t r: #{r}", [:append])
      File.write!("spheregpu.txt", "\t g: #{g}", [:append])
      File.write!("spheregpu.txt", "\t b: #{b}", [:append])
      File.write!("spheregpu.txt", "\n", [:append])
      spherePrinter(list)
    end


    def sphereMaker(spheres, max, max) do
      max = max - 1
        Matrex.set(spheres, 1, max * 7 + 1, Main.rnd(1)) 
        |> Matrex.set( 1, max * 7 + 2, Main.rnd(1)) #g
        |> Matrex.set( 1, max * 7 + 3, Main.rnd(1)) #b
        |> Matrex.set( 1, max * 7 + 4, Main.rnd(20) + 5) #radius
        |> Matrex.set( 1, max * 7 + 5, Main.rnd(Main.dim) - Main.dim/2) #x
        |> Matrex.set( 1, max * 7 + 6, Main.rnd(Main.dim) - Main.dim/2) #y
        |> Matrex.set( 1, max * 7 + 7, Main.rnd(Main.dim) - Main.dim/2) #z
    end
    def sphereMaker(spheres, n, max) do
      
      Matrex.set(spheres, 1, n * 7 + 1, Main.rnd(1)) #r
      |> Matrex.set( 1, (n - 1) * 7 + 2, Main.rnd(1)) #g
      |> Matrex.set( 1, (n - 1) * 7 + 3, Main.rnd(1)) #b
      |> Matrex.set( 1, (n - 1) * 7 + 4, Main.rnd(20) + 5) #radius
      |> Matrex.set( 1, (n - 1) * 7 + 5, Main.rnd(Main.dim) - Main.dim/2) #x
      |> Matrex.set( 1, (n - 1) * 7 + 6, Main.rnd(Main.dim) - Main.dim/2) #y
      |> Matrex.set( 1, (n - 1) * 7 + 7, Main.rnd(Main.dim) - Main.dim/2) #z
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
        sphereList = Matrex.new([sphereMaker2(Main.spheres)])
        spherePrinter(leticia)

        width = Main.dim
        height = Main.dim

        imageList = Matrex.zeros(1, (width) * (height) * 4)

        prev = System.monotonic_time()
        kernel = GPotion.load(&RayTracer.raytracing/3)
        
        refSphere = GPotion.new_gmatrex(sphereList)
        refImag = GPotion.new_gmatrex(imageList)

        GPotion.spawn(kernel,{trunc(width/16),trunc(height/16),1},{16,16,1},[width, refSphere, refImag])
        GPotion.synchronize()
        
        image = GPotion.get_gmatrex(refImag)
        next = System.monotonic_time()



        #------------------file writting----------------------------#
        image = Matrex.to_list(image)
        IO.inspect(image)
        IO.inspect(length(image))

        widthInBytes = width * Bmpgen.bytes_per_pixel


        paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        stride = widthInBytes + paddingSize

        IO.puts("ray tracer completo, começando escrita")
        Bmpgen.writeFileHeader(height, stride)
        Bmpgen.writeInfoHeader(height, width)
        Bmpgen.recursiveWrite(image)

        #-----------------logging times-----------------------------#
        {iteration, _} = Integer.parse(Enum.at(System.argv, 2))
        text = "time: #{next - prev}, iteration: #{iteration}, dimension: #{height}x#{width}, spheres: #{Main.spheres} \n"

        File.write!("time-gpuraytracer.txt", text, [:append])
      
    end
end

Main.main
  


