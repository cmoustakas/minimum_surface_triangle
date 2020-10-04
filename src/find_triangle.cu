/**
  Author : chares moustakas
  
  Running under Geforce 820M. You may need to change SHARED_LIMIT for different GPUs. Have fun.
**/


/** Standard C libraries **/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

/** Standar CUDA libraries **/
#include <cuda.h>
#include <cuda_runtime.h>

#define upper_bound 10000
#define SHARED_LIMIT 16384
#define TOTAL_POINTS 131072

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


typedef struct vec{
    int x;
    int y;
}vec;




int* fill_array(int dim);
double distance(int i,int central_node,int *array,int dim);
bool cross_product(vec vec_1, vec vec_2);
void knn_seq(int *array, int dim, int central_node);
void select_nodes(int *candidates, int *array, int dim, int blocks, int central_node);

__device__ int device_cross_product(int x_1, int y_1, int x_2, int y_2);
__global__ void knn_kernel_distance(size_t pitch,const int* __restrict__ device_array,
                                    double *device_distances, int* device_candidates,
                                    int dim,
                                    int central_node_idx, int central_node_x,int central_node_y,
                                     int numofblocks, int shared_lim_idx,
                                    int offset,int candidates_len);


int main(int argc,char* argv[]){
    if(argc != 3){
        printf("Usage .... \n");
        return -1;
    }

   

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    srand(time(NULL));

    int dim = atoi(argv[1]);
   
    int* array = fill_array(dim);
    int central_node = (rand()%dim) + 1;
    printf("Central_node : %d \n",central_node);

    int *device_array;
    size_t pitch;
    gpuErrchk(cudaMallocPitch(&device_array,&pitch,dim*sizeof(int),2));
    gpuErrchk(cudaMemcpy2D(device_array,pitch,array,dim*sizeof(int),dim*sizeof(int),2,cudaMemcpyHostToDevice));

    const int blocks = atoi(argv[2]);
    int threads;

    int *device_candidates, *candidates;
    int iter_gpu = dim/TOTAL_POINTS;
    size_t candidates_sz;
    int candidates_len;

    if(dim > TOTAL_POINTS){
       
        threads = (int)(TOTAL_POINTS/blocks)+TOTAL_POINTS%blocks;
        candidates_sz = 2*iter_gpu*blocks*sizeof(int);
        candidates_len = blocks*iter_gpu*2;

        gpuErrchk(cudaMalloc(&device_candidates,candidates_sz));
        candidates = (int*)malloc(candidates_sz);
    }
    else{

        threads = (int)(dim/blocks)+dim%blocks;
        candidates_sz = 2*blocks*sizeof(int);
        candidates_len = 2*blocks;


        gpuErrchk(cudaMalloc(&device_candidates,candidates_sz));
        candidates = (int*)malloc(candidates_sz);
    }


    int shared_mem = threads;

    double *device_distances;

    if(dim > SHARED_LIMIT){
        gpuErrchk(cudaMalloc(&device_distances,(TOTAL_POINTS-SHARED_LIMIT)*sizeof(double)));
        shared_mem =  (int)(SHARED_LIMIT/blocks)+SHARED_LIMIT%blocks;
    }
    

    
    
    
    /** 
        Total number of threads: 
        The dimension is 2*dim, so the total host of threads per block 
        must be the number of dim/blocks (division up obviously)
        multiplied by 2. Therefore to total number of threads that will actually run
        is 2 x blocks x dim/blocks = 2 x blocks = host_of_elements.
        Number of blocks must be equal to 8 due to the fact that 2.xx
        capability (Geforce 820M) can actually run in parallel 
        8 blocks of n-times 32 threads.

    **/ 

   



    struct timeval stop_gpu,start_gpu;
    struct timeval stop_cpu,start_cpu;

    /** 
        Start ticking the sequential version
        of KNN algorithm 
    **/
    gettimeofday(&start_cpu,NULL);
    knn_seq(array,dim,central_node);
    gettimeofday(&stop_cpu,NULL);

    
    long unsigned int cpu_t = (stop_cpu.tv_sec - start_cpu.tv_sec)*1000000 + (stop_cpu.tv_usec - start_cpu.tv_usec);
    
    

    /** 
        Start ticking GPU vresion of 
        KNN algorithm
    **/ 
   


    gettimeofday(&start_gpu,NULL);
    
    for(int i=0; i < iter_gpu; i++){
        
        knn_kernel_distance<<<blocks,threads,shared_mem*sizeof(double)>>>(pitch,device_array,device_distances,
                                                                      device_candidates,dim,central_node,array[central_node],
                                                                      array[central_node+dim],blocks,shared_mem,i,candidates_len);
                                                                    
    }                                                              

    gpuErrchk(cudaMemcpy(candidates,device_candidates,candidates_sz,cudaMemcpyDeviceToHost));
    

    select_nodes(candidates,array,dim,candidates_len/2, central_node);
    gettimeofday(&stop_gpu,NULL);

    long unsigned int gpu_t = (stop_gpu.tv_sec - start_gpu.tv_sec)*1000000 + (stop_gpu.tv_usec - start_gpu.tv_usec);


    printf("[+][+][+] CPU PROFILING : %lu us \n\n",cpu_t);

    printf("[+][+][+] GPU PROFILING : %lu us \n\n",gpu_t);

    gpuErrchk(cudaFree(device_candidates));
    gpuErrchk(cudaFree(device_array));
    gpuErrchk(cudaFree(device_distances));


    return 0;

}


__device__ int device_cross_product(int x_1,int y_1,int x_2,int y_2){
    float cos_theta = x_1*x_2 + y_1*y_2/((x_1*x_1 + y_1*y_1) + (x_2*x_2 + y_2*y_2));

    if(cos_theta == 1/((x_1*x_1 + y_1*y_1) + (x_2*x_2 + y_2*y_2)))
        return 0;
    
    return 1;
  
}

__global__ void knn_kernel_distance(size_t pitch,const int* __restrict__ device_array,
                                    double *distances, int* __restrict__ device_candidates,
                                    int dim, int central_node_idx,
                                    int central_node_x,int central_node_y,
                                    int numofblocks, int shared_lim_idx, int offset, int candidates_len){

                                        

    int row = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(row > TOTAL_POINTS){
        return;
    }

    row = row + TOTAL_POINTS*offset;

    int const_x = central_node_x;
    int const_y = central_node_y;
    double d;
    
    extern __shared__  double device_distances[];


    
    int *element_x = (int*)((char*)device_array ) + row;
    int *element_y = (int*)((char*)device_array + pitch) + row;
     
        //__syncthreads();
        

    d = (*element_x -const_x)*(*element_x -const_x);
    d = d +  (*element_y -const_y)*(*element_y -const_y);
   
    if(threadIdx.x < shared_lim_idx)    
        device_distances[threadIdx.x] = d;
    else
        distances[row - TOTAL_POINTS*offset] = d;
        
    

    int k_1 = row;
    int temp_k_2;
    int k_2 = 0 ;
    


    if(threadIdx.x == 0){

        double min_dist = device_distances[threadIdx.x];
        double second_min_dist;

        d = 0;

      
        for(int i=blockDim.x;i--;){
            
            int idx = row + i;
            
            int *element_x = (int*)((char*)device_array ) + idx;
            int *element_y = (int*)((char*)device_array + pitch) + idx;
            
            if(i < shared_lim_idx )
                d = device_distances[i];
            else 
                d = distances[idx - TOTAL_POINTS*offset];

            if(d < min_dist  &&  idx != central_node_idx && *element_x != const_x && *element_y != const_y){

                min_dist = d;
                k_1 = idx;
                
            }

        }

        d = 0;

       
        for(int i=blockDim.x;i--;){
            
            int idx = row + i;
            

            if(i < shared_lim_idx )
                d = device_distances[i];
            else 
                d = distances[idx - TOTAL_POINTS*offset];

            if(d < second_min_dist && d > min_dist){
               
                int *element_x = (int*)((char*)device_array ) + idx;
                int *element_y = (int*)((char*)device_array + pitch) + idx;
            
                temp_k_2 = idx;

                 
                int *element_x_temp = (int*)((char*)device_array ) + temp_k_2;
                int *element_y_temp = (int*)((char*)device_array + pitch) + temp_k_2;

                    /** 
                    Check if the vectors are linear
                    in order to be able to make a triangle
                    **/
             

                if(device_cross_product(*element_x - const_x, // vector_k_1 = [x1,y1]
                   *element_y - const_y,
                   *element_x_temp - const_x, // vector_k_2 = [x2,y2]
                   *element_y_temp-const_y)){
                    
                    k_2 = temp_k_2;
                    second_min_dist = d;
                   
                }

                
            }





        }

        
        device_candidates[blockIdx.x + numofblocks*offset] = k_1;
        device_candidates[candidates_len/2 + blockIdx.x + numofblocks*offset] = k_2;

       


    }

    //__syncthreads();

}



void select_nodes(int *candidates, int *array, int dim, int blocks, int central_node){


    double min_dist = distance(candidates[blocks],central_node,array,dim);
    double sec_min_dist = min_dist;

    int k_1=0;
    int temp_k_2;
    int k_2;
    int k_idx = blocks;


    for(int i=blocks;i--;){
        double dist = distance(candidates[i],central_node,array,dim);
        
        if(min_dist > dist && candidates[i] != central_node ){
            min_dist = dist;
            
            k_1 = candidates[i];
            k_idx = i;

        }
    }

    for(int i=blocks;i--;){
        double dist = distance(candidates[i],central_node,array,dim);
        
        if(dist > min_dist && dist < sec_min_dist && candidates[i]!= central_node){
            temp_k_2 = candidates[i];
              
            vec vec_1;
            vec_1.x = array[temp_k_2] - array[central_node];
            vec_1.y = array[temp_k_2+dim] - array[central_node+dim];

            vec vec_2;
            vec_2.x = array[k_1] - array[central_node];
            vec_2.y = array[k_1+dim] - array[central_node + dim];

            if(cross_product(vec_1,vec_2)){
                k_2 = temp_k_2;
                sec_min_dist = dist;
            }

                        
        }
    }
    
    if(distance(k_2,central_node,array,dim) > distance(candidates[k_idx+blocks],central_node,array,dim))
        k_2 = candidates[k_idx+blocks];

    printf("GPU KNN :  (%d,%d) -- (%d,%d) -- (%d,%d) \n",array[k_1],array[k_1+dim],array[k_2],array[k_2+dim],array[central_node],array[central_node+dim]);


}


double distance(int i,int central_node,int *array,int dim){
    double d = pow(array[i] - array[central_node],2) + pow(array[i+dim] - array[central_node + dim],2);
    //printf("distance(%d) = %lf \n",i,d);

    return d;
}

bool cross_product(vec vec_1, vec vec_2){
    float cos_theta = (vec_1.x*vec_2.x + vec_1.y*vec_2.y)/sqrt( (pow(vec_1.x,2) + pow(vec_1.y,2)) * (pow(vec_2.x,2) + pow(vec_2.y,2))  );
    
    if(cos_theta == 1.f)
        return false;
    
    
    return true;
}



int* fill_array(int dim){

    int* array = (int*)malloc(sizeof(int)*2*dim);
    

    

    for(int rows=0;rows<dim;rows++){
        for(int cols=2;cols--;){
            array[rows + dim*cols] = (rand()%upper_bound) + 1;  
            //printf("(%d,%d) = %d\n",rows,cols,array[rows + dim*cols]);      
        }
    }

    
    return array;

}



void knn_seq(int *array,int dim, int central_node){
    
    double min_dist = distance(0,central_node,array,dim);
    double current_distance;
    int k_1 = 0;
    int k_2;
    int temp_k_2;


    for(int row = 1; row< dim;row++){
        current_distance = distance(row,central_node,array,dim);
        
        if(current_distance < min_dist && row != central_node){
            
            min_dist = current_distance;
            k_1 = row;
        }
            
        
    }




    double sec_min_dist =  distance(0,central_node,array,dim);

    for(int row=1; row<dim; row++){
        current_distance = distance(row,central_node,array,dim);
        
        if(current_distance < sec_min_dist && current_distance > min_dist && row != central_node){
            
            
           
            temp_k_2 = row;
            
            vec vec_1;
            vec_1.x = array[temp_k_2] - array[central_node];
            vec_1.y = array[temp_k_2+dim] - array[central_node+dim];

            vec vec_2;
            vec_2.x = array[k_1] - array[central_node];
            vec_2.y = array[k_1+dim] - array[central_node + dim];

            if(cross_product(vec_1,vec_2)){
                k_2 = temp_k_2;
                sec_min_dist = current_distance;
            }
        }        

    }

    
    printf("CPU KNN : (%d,%d) -- (%d,%d) -- (%d,%d) \n",array[k_1],array[k_1+dim],array[k_2],array[k_2+dim],array[central_node],array[central_node+dim]);


}
