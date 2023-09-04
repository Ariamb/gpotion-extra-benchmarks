#include <stdio.h>

#define DIM 256

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f




//----------------------------------------------------------------


const int BYTES_PER_PIXEL = 4; /// red, green, blue & 255
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);


void generateBitmapImage (unsigned char* image, int height, int width, char* imageFileName){
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE* imageFile = fopen(imageFileName, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 0; i < height; i++) {
        fwrite(image + (i*widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

unsigned char* createBitmapFileHeader (int height, int stride){
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[ 0] = (unsigned char)('B');
    fileHeader[ 1] = (unsigned char)('M');
    fileHeader[ 2] = (unsigned char)(fileSize      );
    fileHeader[ 3] = (unsigned char)(fileSize >>  8);
    fileHeader[ 4] = (unsigned char)(fileSize >> 16);
    fileHeader[ 5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader (int height, int width){
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[ 0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[ 4] = (unsigned char)(width      );
    infoHeader[ 5] = (unsigned char)(width >>  8);
    infoHeader[ 6] = (unsigned char)(width >> 16);
    infoHeader[ 7] = (unsigned char)(width >> 24);
    infoHeader[ 8] = (unsigned char)(height      );
    infoHeader[ 9] = (unsigned char)(height >>  8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL*8);

    return infoHeader;
}






//----------------------------------------------------------------

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        } else {
            return -INF;

        }
    }
};



#define SPHERES 20
__constant__ Sphere s[SPHERES];

__global__ void kernel(int dim, unsigned char *ptr ) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - dim/2);
    float   oy = (y - dim/2);

    float   r=0, g=0, b=0;
    float   maxz = -99999;

    for(int i=0; i<20; i++) {
        float   n;
        float   t = -99999;
        float dx = ox - s[i].x;
        float dy = oy - s[i].y;
        float dz;
        if (dx*dx + dy*dy < s[i].radius * s[i].radius) {
            dz = sqrtf( s[i].radius * s[i].radius - dx*dx - dy*dy );
            n = dz / sqrtf( s[i].radius * s[i].radius );
            t = dz + s[i].z;

        } else {
            t = -99999;
        }
        if (t > maxz) {
              float fscale = n;
              r = s[i].r * fscale;
              g = s[i].g * fscale;
              b = s[i].b * fscale;
              maxz = t;
        }

    }

    //diferença disso:
    //ptr[offset * 4 + 0] = (r * 255);
    //ptr[offset * 4 + 1] = (g * 255);
    //ptr[offset * 4 + 2] = (b * 255);
    //ptr[offset * 4 + 3] = 255;
	
    //pra isso:
	ptr[offset * 4 + 0] = (r * 255);
	ptr[offset * 4 + 1] = (g * 255);
	ptr[offset * 4 + 2] = (b * 255);
    ptr[offset * 4 + 3] = 255;
    
}



int main( void ) {

    //#unsigned char *image; //size: [height][width][BYTES_PER_PIXEL];

    // allocate memory on the GPU for the output bitmap
    unsigned char   *final_bitmap;
    unsigned char   *dev_bitmap;

    cudaMalloc( (void**)&dev_bitmap, 256 * 256 * 4);

    final_bitmap = (unsigned char*) malloc(DIM * DIM *4);
    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    temp_s[0] = {0.5647144993438521,
    0.17026276436658833,
    0.2513199255348369,
    17.309945982238226,
    -83.67052217169714,
    -119.68724631488998,
    98.2803430280465
  };
  temp_s[1]=  {
    0.9091158787804804,
    0.1487777336954863,
    0.1783196508682516,
    21.85598315378277,
    -4.082155827509382,
    -0.5976744895779262,
    24.65309610278635
  };
  temp_s[2]=  {
    0.6624347666859951,
    0.3954588457899716,
    0.6516922513504441,
    17.61146885586108,
    14.65279091769159,
    -110.39790032654805,
    4.207159642323063
  };
  temp_s[3]=  {
    0.413251136814478,
    0.3630481887264626,
    0.1980040894802698,
    16.984618671224098,
    -2.0039674062318795,
    -100.77260658589435,
    -95.8896450697348
  };
  temp_s[4]=  {
    0.13864558854945525,
    0.9300515762810144,
    0.6028931546983245,
    12.94213690603351,
    -104.46021912289804,
    28.098513748588516,
    0.8711203344828675
  };
  temp_s[5]=  {
    0.21469771416364025,
    0.9337748344370861,
    0.33420819727164525,
    18.591723380230107,
    -28.418836024048588,
    107.64000366222115,
    58.74007385479294
  };
  temp_s[6]=  {
    0.576219977416303,
    0.6904812768944365,
    0.7726371044038209,
    18.319498275704213,
    -114.95272682882168,
    88.7097384563738,
    -65.42777794732505
  };
  temp_s[7]=  {
    0.9437543870357372,
    0.3283181249427778,
    0.8446913052766503,
    6.454512161626026,
    122.41389202551346,
    47.942869350260935,
    121.83574938200019
  };
  temp_s[8]=  {
    0.8970305490279855,
    0.014038514358958708,
    0.9583117160557878,
    18.243202002014222,
    15.262184514908284,
    94.37397381511886,
    -126.56245612964263
  };
  temp_s[9]=  {
    0.4650105288857692,
    0.21561326944792017,
    0.8502761925107578,
    24.533677175206762,
    -43.872432630390335,
    -119.06222724082156,
    61.88860744041261
  };
  temp_s[10]=  {
    0.9226660969878231,
    0.9497665334025086,
    0.8874477370525223,
    21.117435224463637,
    -57.17752616962187,
    77.29532761619922,
    -92.29578539384136
  };
  temp_s[11]=  {
    0.03280739768669698,
    0.7397076326792199,
    0.9098178044984283,
    15.11871700186163,
    26.442213202307187,
    16.871608630634483,
    -61.63078707235938
  };
  temp_s[12]=  {
    0.565660573137608,
    0.3304849391155736,
    0.31153294473097937,
    21.61976989043855,
    26.27814569536423,
    -40.46607867671743,
    -1.0898770104068092
  };
  temp_s[13]=  {
    0.14319284646137884,
    0.2749107333597827,
    0.16772972808008058,
    24.909054841761527,
    78.25629444257942,
    10.676107058931251,
    48.06006042664876
  };
  temp_s[14]=  {
    0.007263405255287332,
    0.7207861568041017,
    0.14539017914365063,
    17.106692709128087,
    -84.42054506057924,
    -53.30240791039766,
    114.59334086123235
  };
  temp_s[15]=  {
    0.391155735953856,
    0.3933835871456038,
    0.4371471297341838,
    7.766808069093905,
    123.26548051393169,
    54.50556962797938,
    72.99832148197882
  };
  temp_s[16]=  {
    0.9168065431684317,
    0.9289834284493546,
    0.5631885738700522,
    11.508377330851161,
    -9.691702017273471,
    59.45103305154575,
    -26.8797265541551
  };
  temp_s[17]=  {
    0.06183050019837031,
    0.08331553086947234,
    0.8713950010681478,
    18.9005706961272,
    -13.230872524185912,
    60.95107882930998,
    -63.826166570024725
  };
  temp_s[18]=  {
    0.2659993285927915,
    0.3164159062471389,
    0.46769615771965695,
    15.00518814661092,
    -103.35081026642659,
    -63.951170384838406,
    4.4024781029694395
  };
  temp_s[19]=  {
    0.5646229438154241,
    0.6811426129947813,
    00.023316141239661855,
    14.228797265541552,
    21.32486953337198,
    62.71675771355328,
    -123.35142063661611
  };


    cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES);
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);

    //dim3    grids(DIM,DIM);
    //dim3    threads(1,1);

    kernel<<<grids,threads>>>(256, dev_bitmap );

    // copy our bitmap back from the GPU for display
    cudaMemcpy( final_bitmap, dev_bitmap, DIM * DIM * 4,cudaMemcpyDeviceToHost );




    // get stop time, and display the timing results



    cudaFree( dev_bitmap );

    int height = 256;
    int width = 256;
    unsigned char image[height][width][BYTES_PER_PIXEL];
    char* imageFileName = (char*) "img-craytracer-sanity.bmp";

    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {

              //image[i][j][3] = (unsigned char) 255;                             ///alpha? 255
              //image[i][j][2] = (unsigned char)  ( i * 255 / height );             ///red//blue
              //image[i][j][1] = (unsigned char) ( j * 255 / width );              ///green//green
              //image[i][j][0] = (unsigned char) ( (i+j) * 255 / (height+width) ); ///blue // final_bitmap[red]

            image[i][j][3] = (unsigned char) final_bitmap[(i * 256 + j) * 4 + 3] ;
            image[i][j][0] = (unsigned char) final_bitmap[(i * 256 + j) * 4 + 2] ;
            image[i][j][1] = (unsigned char) final_bitmap[(i * 256 + j) * 4 + 1] ;
            image[i][j][2] = (unsigned char) final_bitmap[(i * 256 + j) * 4 + 0] ;
        }
    }

    generateBitmapImage((unsigned char*) image, height, width, imageFileName);
    printf("Image generated!!");

    //no display, only write to file
    printf("final bmp! %d %d %d %d \n", final_bitmap[0], final_bitmap[1], final_bitmap[2], final_bitmap[3]);
    printf("final bmp! %d %d %d %d \n", final_bitmap[4], final_bitmap[5], final_bitmap[6], final_bitmap[7]);

    printf("img! %d %d %d %d \n", image[0][0][0], image[0][0][1], image[0][0][2], image[0][0][3]);
    printf("img! %d %d %d %d \n", image[0][1][0], image[0][1][1], image[0][1][2], image[0][1][3]);


}
