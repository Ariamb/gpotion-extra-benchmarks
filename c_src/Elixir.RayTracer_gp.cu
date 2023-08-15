#include "erl_nif.h"

__global__
void raytracing(int dim, float *spheres, float *image)
{
	int x = (threadIdx.x + (blockIdx.x * blockDim.x));
	int y = (threadIdx.y + (blockIdx.y * blockDim.y));
	int offset = (x + ((y * blockDim.x) * gridDim.x));
	int ox = (x - (dim / 2));
	int oy = (y - (dim / 2));
	float r = 0.0;
	float g = 0.0;
	float b = 0.0;
	float maxz = (- 9999999.0);
for( int i = 0; i<20; i++){
	float n = 0.0;
	float dx = (ox - spheres[((i * 7) + 4)]);
	float dy = (oy - spheres[((i * 7) + 5)]);
if((((dx * dx) + (dy * dy)) < (spheres[((i * 7) + 3)] * spheres[((i * 7) + 3)])))
{
	float dzsqrd = (((spheres[((i * 7) + 3)] * spheres[((i * 7) + 3)]) - (dx * dx)) - (dy * dy));
	float dz = sqrt(dzsqrd);
	n = (dz / sqrt((spheres[((i * 7) + 3)] * spheres[((i * 7) + 3)])));
	dz = (dz + spheres[((i * 7) + 6)]);
}

	float fscale = 0.0;
if((dz > maxz))
{
	fscale = n;
	r = (spheres[((i * 7) + 0)] * fscale);
	g = (spheres[((i * 7) + 1)] * fscale);
	b = (spheres[((i * 7) + 2)] * fscale);
	maxz = dz;
}

}

	image[((offset * 4) + 0)] = (r * 255);
	image[((offset * 4) + 1)] = (g * 255);
	image[((offset * 4) + 2)] = (b * 255);
	image[((offset * 4) + 3)] = 255;
}

extern "C" void raytracing_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;
    float **array_res;

    const ERL_NIF_TERM *tuple_blocks;
    const ERL_NIF_TERM *tuple_threads;
    int arity;

    if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
      printf ("spawn: blocks argument is not a tuple");
    }

    if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
      printf ("spawn:threads argument is not a tuple");
    }
    int b1,b2,b3,t1,t2,t3;

    enif_get_int(env,tuple_blocks[0],&b1);
    enif_get_int(env,tuple_blocks[1],&b2);
    enif_get_int(env,tuple_blocks[2],&b3);
    enif_get_int(env,tuple_threads[0],&t1);
    enif_get_int(env,tuple_threads[1],&t2);
    enif_get_int(env,tuple_threads[2],&t3);

    dim3 blocks(b1,b2,b3);
    dim3 threads(t1,t2,t3);

    list= argv[3];

  enif_get_list_cell(env,list,&head,&tail);
  int arg1;
  enif_get_int(env, head, &arg1);
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg2 = *array_res;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg3 = *array_res;
  list = tail;

   raytracing<<<blocks, threads>>>(arg1,arg2,arg3);
    cudaError_t error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)
     { char message[200];
       strcpy(message,"Error kernel call: ");
       strcat(message, cudaGetErrorString(error_gpu));
       enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
     }
}
