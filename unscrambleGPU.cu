////////////////////////////////////////ECE 406 Final Project///////////////////////////////////////////////////
//                                      GPU Version by Renfei Wang and Shaowei Su                             //
//                                      This program will unscramble the image                                //
//                                      Please input three arguments                                          //
//                                      <image filename> <csv filename> <Box size>                            //
//                                      the BoxSize is 2,4 or 8 according to the csv file                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>


using namespace cv;

int M;  // number of rows in image
int N;  // number of columns in image
int numBox;
int boxSize;
int box_col; // equals to the box_row

cudaError_t launch_unscramble(uchar *p,uint64_t *csvMat,int boxSize,int box_col,int *result_matrix_row,int *result_matrix_col,int *result_xor_row,int *result_xor_col,int M,float* Runtimes,int *row_xor,int *col_xor);


__global__ void unscramble_kernel(uchar *GPU_image,uint64_t *GPU_csvMat,int boxSize,int box_col,int M,int *GPU_result_matrix_row,int *GPU_result_matrix_col,int *GPU_result_xor_row,int *GPU_result_xor_col,int *GPU_row_xor,int *GPU_col_xor){
    
    int x = blockIdx.x;
    int i=  threadIdx.x; 
    int k = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t temp1,temp2,temp3;
    int result; 

    extern __shared__ int sdata_row[];
    extern __shared__ int sdata_col[];
///////////This is calculate the xor of every row in the checkbox/////////////////
    if(i<box_col){   
       temp1 = GPU_csvMat[i*2+1+x*box_col*2];
      for(k=0;k<boxSize;k++){
        result=0;
        temp2 = temp1>>8;
        temp3 = temp2<<8;
        result = temp1 - temp3;
        temp1 = GPU_csvMat[i*2+1+x*box_col*2]>>8*(k+1);
        GPU_result_matrix_row[(boxSize-1-k)*box_col+i+boxSize*x*box_col]=result;
      }    
    }
///////////This is calculate the xor of every column in the checkbox/////////////////
    if(i>=box_col&&i<2*box_col){
      temp1 = GPU_csvMat[(i-box_col)*2+x*box_col*2];
      for(k=0;k<boxSize;k++){
        result=0;
        temp2 = temp1>>8;
        temp3 = temp2<<8;
        result = temp1 - temp3;
        temp1 = GPU_csvMat[(i-box_col)*2+x*box_col*2]>>8*(k+1);        
        GPU_result_matrix_col[(i-box_col)*box_col*boxSize+x+(boxSize-1-k)*box_col]=result;
        }  
    }    
    __syncthreads();   
///////////////This is the xor of each row////////////////////////////////////////
    if(tid<M){
        GPU_result_xor_row[tid]=GPU_result_matrix_row[tid*box_col];
        for(k=1;k<box_col;k++){ 
          GPU_result_xor_row[tid]=GPU_result_xor_row[tid]^GPU_result_matrix_row[tid*box_col+k];
      }  
    }  
///////////////This is the xor of each column////////////////////////////////////////    
    if(tid>=M&&tid<2*M){
      for(k=0;k<box_col-1;k++){
        GPU_result_matrix_col[k+1+(tid-M)*box_col]=GPU_result_matrix_col[k+(tid-M)*box_col]^GPU_result_matrix_col[k+1+(tid-M)*box_col];
      }
      GPU_result_xor_col[tid-M]=GPU_result_matrix_col[box_col-1+(tid-M)*box_col];
    }  
/////////////This is to calculate the xor of row in the scramble image///////////////
    if(tid>=2*M&&tid<3*M){
      sdata_row[tid-2*M] = GPU_image[(tid-2*M)*M];
      for(k=0;k<M-1;k++){
          sdata_row[tid-2*M]=sdata_row[tid-2*M]^GPU_image[(tid-2*M)*M+k+1];
      }
      GPU_row_xor[tid-2*M]=sdata_row[tid-2*M];
    }
/////////////This is to calculate the xor of column in the scramble image///////////////
    if(tid>=3*M&&tid<4*M){
      sdata_col[tid-3*M] = GPU_image[tid-3*M];
      for(k=1;k<M;k++){
          sdata_col[tid-3*M]=sdata_col[tid-3*M]^GPU_image[k*M+tid-3*M];
      }
      GPU_col_xor[tid-3*M]=sdata_col[tid-3*M];
    }
    __syncthreads();
}

int main(int argc, char *argv[]){
	int i, j;
	int *row_xor, *col_xor;

  float GPURuntimes[4];   // run times of the GPU code
  cudaError_t cudaStatus;

	if( argc != 4) {
		printf("Usage: input format: <image filename><csv filename><Box size>\n");
		printf("box size should be 2, 4 or 8\n");
		exit(EXIT_FAILURE);
	}

/////////////////////image load/////////////////////////////////////////
	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);//read the image
	if(! image.data ) {
		fprintf(stderr, "Could not open the image.\n");
		exit(EXIT_FAILURE);
	}
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

	// Set up global variables based on image size:
	M = image.rows;
	N = image.cols;
	boxSize = atoi(argv[3]);
	numBox = pow(M / boxSize, 2);
	box_col= M/boxSize;// how many box in one col

///////////////////malloc memory for the xor////////////////////////////
	row_xor = (int*) malloc(M*sizeof(int));
	if(row_xor == NULL){ printf("Fail to melloc \n\n"); exit(EXIT_FAILURE); }
	col_xor = (int*) malloc(N*sizeof(int));
	if(col_xor == NULL){ printf("Fail to melloc \n\n"); exit(EXIT_FAILURE); }

	uchar *p = image.data;

    char buffer[1024] ;
    char *record,*line;
    i = 0;
    j = 0;
    uint64_t csvmat_read[numBox][2];
    uint64_t csvMat[numBox*2];

/////////////////csv file load/////////////////////////////////////////
    FILE *fstream = fopen(argv[2],"r");
    if(fstream == NULL)
    {
       	printf("\n file opening failed ");
       	exit(EXIT_FAILURE);
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
    {	
    	j=0;
      	record = strtok(line,",");
      	while(record != NULL)
      	{ 
      		csvmat_read[i][j] = strtoull(record,0,0) ;
      		//printf("record : %lld at %d, %d \n", csvmat_read[i][j], i, j) ; 
      		record = strtok(NULL,",");
      		j++;
      	}
      	++i ;
   }

   for(int i=0;i<numBox;i=i+1){
   		csvMat[2*i]=csvmat_read[i][0];
   		csvMat[2*i+1]=csvmat_read[i][1];
   }

////////////some varibles and memories malloc///////////////////
   int *result_matrix_row;	
   int *result_matrix_col;
   int *result_xor_row;
   int *result_xor_col;

   result_matrix_row= (int*) malloc(M*box_col*sizeof(int));// this is to store the decimal which is transformed from the 8 digits
   if(result_matrix_row == NULL){ printf("Fail to melloc result_matrix_row\n\n"); exit(EXIT_FAILURE); }

   result_matrix_col= (int*) malloc(M*box_col*sizeof(int));// this is to store the decimal which is transformed from the 8 digits
   if(result_matrix_col == NULL){ printf("Fail to melloc result_matrix_col\n\n"); exit(EXIT_FAILURE); }

   result_xor_row= (int*) malloc(M*sizeof(int));
   if(result_xor_row == NULL){ printf("Fail to melloc result_xor_row\n\n"); exit(EXIT_FAILURE); }

   result_xor_col= (int*) malloc(M*sizeof(int));
   if(result_xor_col == NULL){ printf("Fail to melloc result_xor_col\n\n"); exit(EXIT_FAILURE); }

   uchar *temp_image;
   temp_image=(uchar*) malloc(M*N*sizeof(uchar));
   if(temp_image == NULL){ printf("Fail to melloc p\n\n"); exit(EXIT_FAILURE); }

   Mat temp = Mat(M, N, CV_8UC1, temp_image);


/////////////////////launch the GPU part////////////////////////
  cudaStatus = launch_unscramble(p,csvMat,boxSize,box_col,result_matrix_row,result_matrix_col,result_xor_row,result_xor_col,M,GPURuntimes,row_xor,col_xor);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "launch_unscramble failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("-----------------------------------------------------------------\n");
  printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
      GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
  printf("-----------------------------------------------------------------\n");  

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!\n");
    exit(EXIT_FAILURE);
  } 

////////////////get the unscramble image, first by row//////////////////////////////////
   int flag1=0;
   int flag2=0;
   int swap[256];
   for(int i=0;i<256;i++){
   		swap[i]=0;
   }
   	for(int j=0;j<N;j++){
   		for(int i=0;i<M;i++){//swap from this line
   			if(result_xor_row[j]==row_xor[i] && swap[i]==0){// if find the targets, then swap	
   				swap[i]=1;
   				Mat M1 = temp.row(j);
   				image.row(i).copyTo(M1);
   				flag1++;
   				//printf("has swaped column %d and %d and the result_xor is %d the row_xor is %d\n",j,i,result_xor[j],row_xor[i]);
   				break;
   			}
   		}
   	}	
////////////////get the unscramble image, then by column//////////////////////////////////
    for(int i=0;i<256;i++){
   		swap[i]=0;
    }	
   	for(int j=0;j<N;j++){
   		for(int i=0;i<M;i++){//swap from this line
   			if(result_xor_col[j]==col_xor[i] && swap[i]==0){// if find the targets, then swap
   				swap[i]=1;
   				flag2++;
   				Mat M2 = image.col(j);
   				temp.col(i).copyTo(M2);
   				//printf("has swaped row %d and %dand the result_xor is %d the row_xor is %d\n",j,i,result_xor[j],col_xor[i]);
   				break;
   			}
   		}
   	}	
   	printf("%d, %d \n",flag1,flag2);

	// Display the output image:
	Mat result = Mat(M, N, CV_8UC1, image.data);
	// and save it to disk:
	string output_filename = "unscramble.png";
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		exit(EXIT_FAILURE);
	}

	printf("Saved image '%s', size = %dx%d (dims = %d).\n", output_filename.c_str(), result.rows, result.cols, result.dims);

	free(row_xor);
	free(col_xor);
	free(result_matrix_row);
  free(result_matrix_col);
	free(result_xor_row);
  free(result_xor_col);  
	free(temp_image);
  exit(EXIT_SUCCESS);
}

// Helper function for launching a CUDA kernel (including memcpy, timing, etc.):
cudaError_t launch_unscramble(uchar *p,uint64_t *csvMat,int boxSize,int box_col,int *result_matrix_row,int *result_matrix_col,int *result_xor_row,int *result_xor_col,int M,float* Runtimes,int *row_xor,int *col_xor)
{
  cudaEvent_t time1, time2, time3, time4;
  uint64_t *GPU_csvMat;
  int *GPU_result_matrix_row;
  int *GPU_result_matrix_col;
  int *GPU_result_xor_row;
  int *GPU_result_xor_col;
  uchar *GPU_image;
  int *GPU_row_xor;
  int *GPU_col_xor;

  // Choose which GPU to run on; change this on a multi-GPU system.
  cudaError_t cudaStatus;
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    goto Error;
  }

  cudaEventCreate(&time1);
  cudaEventCreate(&time2);
  cudaEventCreate(&time3);
  cudaEventCreate(&time4);

  cudaEventRecord(time1, 0);

  // Allocate GPU buffer for inputs and outputs:
  cudaStatus = cudaMalloc((void**)&GPU_image, M*N*sizeof(uchar));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_image cudaMalloc failed!\n");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&GPU_csvMat, 2*box_col*box_col*sizeof(uint64_t));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_csvMat cudaMalloc failed!\n");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&GPU_result_matrix_row, M*box_col*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_result_matrix_row cudaMalloc failed!\n");
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&GPU_result_matrix_col, M*box_col*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_result_matrix_col cudaMalloc failed!\n");
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&GPU_result_xor_row, M*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_result_xor_row cudaMalloc failed!\n");
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&GPU_result_xor_col, M*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_result_xor_col cudaMalloc failed!\n");
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&GPU_row_xor, M*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_row_xor cudaMalloc failed!\n");
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&GPU_col_xor, M*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_col_xor cudaMalloc failed!\n");
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(GPU_csvMat, csvMat, 2*box_col*box_col*sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_csvMat cudaMemcpy failed!\n");
    goto Error;
  }
  cudaStatus = cudaMemcpy(GPU_image, p, M*N*sizeof(uchar), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_csvMat cudaMemcpy failed!\n");
    goto Error;
  }

  cudaEventRecord(time2, 0);

  // Launch a kernel on the GPU 
  unscramble_kernel<<<box_col,M,2*M*sizeof(int)>>>(GPU_image,GPU_csvMat,boxSize,box_col,M,GPU_result_matrix_row,GPU_result_matrix_col,GPU_result_xor_row,GPU_result_xor_col,GPU_row_xor,GPU_col_xor);
  // Check for errors immediately after kernel launch.
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaEventRecord(time3, 0);

  // Copy output (results) from GPU buffer to host (CPU) memory.
 cudaStatus = cudaMemcpy(result_xor_row, GPU_result_xor_row,  M*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "result_xor_row cudaMemcpy failed!\n");
    goto Error;
  }
 
 cudaStatus = cudaMemcpy(result_xor_col, GPU_result_xor_col,  M*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "result_xor_row cudaMemcpy failed!\n");
    goto Error;
  }  
  cudaStatus = cudaMemcpy(row_xor, GPU_row_xor,  M*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "row_xor cudaMemcpy failed!\n");
    goto Error;
  }
 cudaStatus = cudaMemcpy(col_xor, GPU_col_xor,   M*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "row_xor cudaMemcpy failed!\n");
    goto Error;
  }   
  
  cudaStatus = cudaMemcpy(result_matrix_row, GPU_result_matrix_row,   M*box_col*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "result_xor_row cudaMemcpy failed!\n");
    goto Error;
  }
 cudaStatus = cudaMemcpy(result_matrix_col, GPU_result_matrix_col,   M*box_col*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "result_xor_row cudaMemcpy failed!\n");
    goto Error;
  }  

  cudaEventRecord(time4, 0);
  cudaEventSynchronize(time1);
  cudaEventSynchronize(time2);
  cudaEventSynchronize(time3);
  cudaEventSynchronize(time4);

  float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;

  cudaEventElapsedTime(&totalTime, time1, time4);
  cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
  cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
  cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

  Runtimes[0] = totalTime;
  Runtimes[1] = tfrCPUtoGPU;
  Runtimes[2] = kernelExecutionTime;
  Runtimes[3] = tfrGPUtoCPU;

  Error:
  cudaFree(GPU_csvMat);
  cudaFree(GPU_result_matrix_row);
  cudaFree(GPU_result_matrix_col);
  cudaFree(GPU_result_xor_row);
  cudaFree(GPU_result_xor_col);
  cudaFree(GPU_image);
  cudaFree(GPU_row_xor);
  cudaFree(GPU_col_xor);
  cudaEventDestroy(time1);
  cudaEventDestroy(time2);
  cudaEventDestroy(time3);
  cudaEventDestroy(time4);

  return cudaStatus;
}
