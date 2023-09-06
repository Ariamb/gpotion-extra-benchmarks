#include <stdio.h>

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f




//----------------------------------------------------------------


const int BYTES_PER_PIXEL = 4; /// red, green, blue & 255
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);


void generateLog(double time, int spheres, int interation){
  printf("Writting operation time to file.\n");
  FILE *file;
  char filename[] = "time-c-cpuraytracer.txt";

  file = fopen(filename, "a");
  fprintf(file, "time: %f \t spheres %d \t iteration %d.\n", time, spheres, interation);
  fclose(file);
  
  printf("done.\n");


}

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

__global__ void kernel(int dim,  unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - dim/2);
    float   oy = (y - dim/2);

    float   r=0, g=0, b=0;
    float   maxz = -99999;
    for(int i=0; i<SPHERES; i++) {
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

    ptr[offset*4 + 0] = (r * 255);
    ptr[offset*4 + 1] = (g * 255);
    ptr[offset*4 + 2] = (b * 255);
    ptr[offset*4 + 3] = 255;
}




int main(int argc, char *argv[]){
    int dim = atoi(argv[1]);
    int sph = atoi(argv[2]);
    int iteration = atoi(argv[3]);


    // allocate memory on the GPU for the output bitmap
    unsigned char   *final_bitmap;
    unsigned char   *dev_bitmap;

    cudaMalloc( (void**)&dev_bitmap, dim * dim * 4);

    final_bitmap = (unsigned char*) malloc(dim * dim *4);

    if (!final_bitmap) { perror("malloc arr"); exit(EXIT_FAILURE); };
    //if (!temp_s) { perror("malloc arr"); exit(EXIT_FAILURE); };
    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );

    /*
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 256.0f ) - 128;
        temp_s[i].y = rnd( 256.0f ) - 128;
        temp_s[i].z = rnd( 256.0f ) - 128;
        temp_s[i].radius = rnd( 20.0f ) + 5;
        printf("sphere{\n");
        printf("temp_[] = { %f \n", temp_s[i].r);}
        printf("g: %f \n", temp_s[i].g);
        printf("b: %f \n", temp_s[i].b);
        printf("radius: %f \n", temp_s[i].radius);
        printf("x: %f \n", temp_s[i].x);
        printf("y: %f \n", temp_s[i].y);
        printf("z: %f \n", temp_s[i].z);
    }
*/

    if(dim == 256) {
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
    }
    if(dim == 1024){
      temp_s[0] = { 0.5647144993438521, 0.17026276436658833, 0.2513199255348369, 69.2397839289529, -334.6820886867886, -478.7489852595599, 98.2803430280465};
      temp_s[1] = { 0.9091158787804804, 0.1487777336954863, 0.1783196508682516, 87.42393261513108, -16.32862331003753, -2.390697958311705, 24.65309610278635};
      temp_s[2] = { 0.6624347666859951, 0.3954588457899716, 0.6516922513504441, 70.44587542344432, 58.61116367076636, -441.5916013061922, 4.207159642323063};
      temp_s[3] = { 0.413251136814478, 0.3630481887264626, 0.1980040894802698, 67.93847468489639, -8.015869624927518, -403.0904263435774, -95.8896450697348};
      temp_s[4] = { 0.13864558854945525, 0.9300515762810144, 0.6028931546983245, 51.76854762413404, -417.84087649159216, 112.39405499435406, 0.8711203344828675};
      temp_s[5] = { 0.21469771416364025, 0.9337748344370861, 0.33420819727164525, 74.36689352092043, -113.67534409619435, 430.5600146488846, 58.74007385479294};
      temp_s[6] = { 0.576219977416303, 0.6904812768944365, 0.7726371044038209, 73.27799310281685, -459.8109073152867, 354.8389538254952, -65.42777794732505};
      temp_s[7] = { 0.9437543870357372, 0.3283181249427778, 0.8446913052766503, 25.818048646504103, 489.65556810205385, 191.77147740104374, 121.83574938200019};
      temp_s[8] = { 0.8970305490279855, 0.014038514358958708, 0.9583117160557878, 72.97280800805689, 61.04873805963314, 377.49589526047544, -126.56245612964263};
      temp_s[9] = { 0.4650105288857692, 0.21561326944792017, 0.8502761925107578, 98.13470870082705, -175.48973052156134, -476.2489089632862, 61.88860744041261};
      temp_s[10] = { 0.9226660969878231, 0.9497665334025086, 0.8874477370525223, 84.46974089785455, -228.71010467848748, 309.18131046479687, -92.29578539384136};
      temp_s[11] = { 0.03280739768669698, 0.7397076326792199, 0.9098178044984283, 60.47486800744652, 105.76885280922875, 67.48643452253793, -61.63078707235938};
      temp_s[12] = { 0.565660573137608, 0.3304849391155736, 0.31153294473097937, 86.4790795617542, 105.11258278145692, -161.86431470686972, -1.0898770104068092};
      temp_s[13] = { 0.14319284646137884, 0.2749107333597827, 0.16772972808008058, 99.63621936704611, 313.02517777031767, 42.704428235725004, 48.06006042664876};
      temp_s[14] = { 0.007263405255287332, 0.7207861568041017, 0.14539017914365063, 68.42677083651235, -337.682180242317, -213.20963164159065, 114.59334086123235};
      temp_s[15] = { 0.391155735953856, 0.3933835871456038, 0.4371471297341838, 31.06723227637562, 493.06192205572677, 218.02227851191753, 72.99832148197882};
      temp_s[16] = { 0.9168065431684317, 0.9289834284493546, 0.5631885738700522, 46.033509323404644, -38.766808069093884, 237.804132206183, -26.8797265541551};
      temp_s[17] = { 0.06183050019837031, 0.08331553086947234, 0.8713950010681478, 75.6022827845088, -52.92349009674365, 243.8043153172399, -63.826166570024725};
      temp_s[18] = { 0.2659993285927915, 0.3164159062471389, 0.46769615771965695, 60.02075258644368, -413.40324106570637, -255.80468153935362, 4.4024781029694395};
      temp_s[19] = { 0.5646229438154241, 0.6811426129947813, 0.023316141239661855, 56.915189062166206, 85.29947813348792, 250.8670308542131, -123.35142063661611};
    }
    if(dim == 3096){
      temp_s[0]={ 0.5647144993438521, 0.17026276436658833, 0.2513199255348369, 118.47956785790582, -1011.8903775139622, -1447.4676351207006, 98.2803430280465};
      temp_s[1]={ 0.9091158787804804, 0.1487777336954863, 0.1783196508682516, 154.84786523026216, -49.36857203894169, -7.228125858332987, 24.65309610278635};
      temp_s[2]={ 0.6624347666859951, 0.3954588457899716, 0.6516922513504441, 120.89175084688864, 177.20719016083262, -1335.1246070741904, 4.207159642323063};
      temp_s[3]={ 0.413251136814478, 0.3630481887264626, 0.1980040894802698, 115.87694936979278, -24.2354808191169, -1218.7187108981598, -95.8896450697348};
      temp_s[4]={ 0.13864558854945525, 0.9300515762810144, 0.6028931546983245, 83.53709524826807, -1263.3157750175483, 339.8164006469924, 0.8711203344828675};
      temp_s[5]={ 0.21469771416364025, 0.9337748344370861, 0.33420819727164525, 128.73378704184086, -343.6902981658375, 1301.771294289987, 58.74007385479294};
      temp_s[6]={ 0.576219977416303, 0.6904812768944365, 0.7726371044038209, 126.55598620563372, -1390.2095400860621, 1072.8333994567706, -65.42777794732505};
      temp_s[7]={ 0.9437543870357372, 0.3283181249427778, 0.8446913052766503, 31.63609729300821, 1480.4430066835534, 579.8090762047182, 121.83574938200019};
      temp_s[8]={ 0.8970305490279855, 0.014038514358958708, 0.9583117160557878, 125.94561601611377, 184.57704397717225, 1141.335245826594, -126.56245612964263};
      temp_s[9]={ 0.4650105288857692, 0.21561326944792017, 0.8502761925107578, 176.2694174016541, -530.5822321237831, -1439.9088106936856, 61.88860744041261};
      temp_s[10]={ 0.9226660969878231, 0.9497665334025086, 0.8874477370525223, 148.9394817957091, -691.4907071138646, 934.7903683584095, -92.29578539384136};
      temp_s[11]={ 0.03280739768669698, 0.7397076326792199, 0.9098178044984283, 100.94973601489303, 319.7855159154026, 204.04101687673574, -61.63078707235938};
      temp_s[12]={ 0.565660573137608, 0.3304849391155736, 0.31153294473097937, 152.9581591235084, 317.8013245033112, -489.3866389965515, -1.0898770104068092};
      temp_s[13]={ 0.14319284646137884, 0.2749107333597827, 0.16772972808008058, 179.27243873409222, 946.412060914945, 129.1141697439498, 48.06006042664876};
      temp_s[14]={ 0.007263405255287332, 0.7207861568041017, 0.14539017914365063, 116.8535416730247, -1020.9609668263802, -644.6259956663716, 114.59334086123235};
      temp_s[15]={ 0.391155735953856, 0.3933835871456038, 0.4371471297341838, 42.13446455275124, 1490.7419049653613, 659.1767326883755, 72.99832148197882};
      temp_s[16]={ 0.9168065431684317, 0.9289834284493546, 0.5631885738700522, 72.06701864680929, -117.20902127140107, 718.9859309671315, -26.8797265541551};
      temp_s[17]={ 0.06183050019837031, 0.08331553086947234, 0.8713950010681478, 131.2045655690176, -160.01086458937357, 737.1271095919674, -63.826166570024725};
      temp_s[18]={ 0.2659993285927915, 0.3164159062471389, 0.46769615771965695, 100.04150517288736, -1249.8988616595966, -773.4094668416394, 4.4024781029694395};
      temp_s[19]={ 0.5646229438154241, 0.6811426129947813, 0.023316141239661855, 93.83037812433241, 257.89764091921757, 758.480788598285, -123.35142063661611};
    }
//*/

    cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES);
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(64,64);
    dim3    threads(16,16);

    //dim3    grids(DIM,DIM);
    //dim3    threads(1,1);

    kernel<<<grids,threads>>>(dim, dev_bitmap );

    // copy our bitmap back from the GPU for display
    cudaMemcpy( final_bitmap, dev_bitmap, dim * dim * 4,cudaMemcpyDeviceToHost );


    // get stop time, and display the timing results



    cudaFree( dev_bitmap );

    int height = dim;
    int width = dim;
    unsigned char* image = (unsigned char*) malloc(dim * dim *4); //[height][width][BYTES_PER_PIXEL];
    char* imageFileName = (char*) "bitmapImage.bmp";

    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            image[(i * dim + j) * 4 + 3] = final_bitmap[(i * dim + j) * 4 + 3] ;
            image[(i * dim + j) * 4 + 0] = final_bitmap[(i * dim + j) * 4 + 2] ;
            image[(i * dim + j) * 4 + 1] = final_bitmap[(i * dim + j) * 4 + 1] ;
            image[(i * dim + j) * 4 + 2] = final_bitmap[(i * dim + j) * 4 + 0] ;
        }
    }

    generateBitmapImage((unsigned char*) image, height, width, imageFileName);
    printf("Image generated!!");

    generateLog(dim, sph, iteration);
    //no display, only write to file

    free(image);
    free(final_bitmap);
W
}
