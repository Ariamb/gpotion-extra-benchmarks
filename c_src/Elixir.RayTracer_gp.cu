#include "erl_nif.h"

__global__
void raytracing(int width, int height, float *spheres, float *image)
{
	int x = (threadIdx.x + (blockIdx.x * blockDim.x));
	int y = (threadIdx.y + (blockIdx.y * blockDim.y));
	int offset = (x + ((y * blockDim.x) * gridDim.x));
	float ox = 0.0;
	float oy = 0.0;
	ox = (x - (width / 2));
	oy = (y - (height / 2));
	float r = 0.0;
	float g = 0.0;
	float b = 0.0;
	float maxz = (- 99999.0);
for( int i = 0; i<20; i++){
	float sphereRadius = spheres[((i * 7) + 3)];
	float dx = (ox - spheres[((i * 7) + 4)]);
	float dy = (oy - spheres[((i * 7) + 5)]);
	float n = 0.0;
	float t = (- 99999.0);
	float dz = 0.0;
if((((dx * dx) + (dy * dy)) < (sphereRadius * sphereRadius)))
{
	dz = sqrtf((((sphereRadius * sphereRadius) - (dx * dx)) - (dy * dy)));
	n = (dz / sqrtf((sphereRadius * sphereRadius)));
	t = (dz + spheres[((i * 7) + 6)]);
}
else{
	t = (- 99999.0);
	n = 0.0;
}

if((t > maxz))
{
	float fscale = n;
	r = (spheres[((i * 7) + 0)] * fscale);
	g = (spheres[((i * 7) + 1)] * fscale);
	b = (spheres[((i * 7) + 2)] * fscale);
	maxz = t;
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
  int arg2;
  enif_get_int(env, head, &arg2);
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg3 = *array_res;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg4 = *array_res;
  list = tail;

   raytracing<<<blocks, threads>>>(arg1,arg2,arg3,arg4);
    cudaError_t error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)
     { char message[200];
       strcpy(message,"Error kernel call: ");
       strcat(message, cudaGetErrorString(error_gpu));
       enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
     }
}
