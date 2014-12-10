#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
//#include <iostream>
//#include <sstream>
#include <sys/time.h>


using namespace cv;

int M;  // number of rows in image
int N;  // number of columns in image
int numBox;
int boxSize;
int box_col; // equals to the box_row

cudaError_t launch_unscramble(uchar *p,uint64_t *csvMat,int boxSize,int box_col,int *binary,int *result_matrix_row,int *result_matrix_col,int bits_size,int *result_xor_row,int *result_xor_col,int M,float* Runtimes);

//decimal to binary function
__device__ void decimal_to_binary(uint64_t decimal, int bits_size, int *binary){
	int digit;
  int bits_input = bits_size;
	int n=bits_input-1;
	  for (bits_input=n; bits_input >= 0; bits_input--)
	  {
	    digit = decimal >> bits_input;// csv_number is the number	 
	    if (digit & 1){
		  binary[n-bits_input]=1;
		}
	    else{
		  binary[n-bits_input]=0;
		}
	  }			
}

__device__ int power(int pow){
  int result=1;
  for(int i=0;i<pow;i++){
    result=result*2;
  }
  return result;
}

__global__ void unscramble_kernel(uint64_t *GPU_csvMat,int boxSize,int box_col,int M,int *GPU_binary,int *GPU_result_matrix_row,int *GPU_result_matrix_col,int bits_size){
    int x = blockIdx.x;
    int i=  threadIdx.x; 
    //int id = blockIdx.x * blockDim.x + threadIdx.x;
    decimal_to_binary(GPU_csvMat[i*2+1+x*box_col*2],bits_size,GPU_binary);
    __syncthreads();

    for(int k=0;k<boxSize;k++){
        int result=0;
        for(int z=0;z<8;z++){//change the binary to the decimal to XOR
            result=result+GPU_binary[z+k*8]*power(7-z);
            //printf("result is:%d\n",result);
          }
          GPU_result_matrix_row[k*box_col+i+boxSize*x*box_col]=result;
          //printf("checkbox row result %d: result_matrix[%d] = %d\n",x*256+8*i+k,k*box_col+i+boxSize*x*box_col,result_matrix[k*box_col+i+boxSize*x*box_col]);
        }
    __syncthreads();

    decimal_to_binary(GPU_csvMat[i*2+x*box_col*2],bits_size,GPU_binary);//i*2+x*box_col*2//x*2+i*box_col*2
    __syncthreads();

    for(int k=0;k<boxSize;k++){
        int result=0;
        for(int z=0;z<8;z++){//change the binary to the decimal to XOR
            result=result+GPU_binary[z+k*8]*power(7-z);
          }
          GPU_result_matrix_col[i*box_col*boxSize+x+k*box_col]=result;//i*box_col*boxSize+x+k*box_col//k*box_col+i+boxSize*x*box_col
          //printf("checkbox column result %d: %d\n",x*256+boxSize*i+k,result_matrix[k*box_col+i+boxSize*x*box_col]);
        }
    __syncthreads();
    
}

/*
void checkbox_binary_row(uint64_t *csvMat,int boxSize,int box_col,int *binary,int *result_matrix,int bits_size){
	for(int x=0;x<box_col;x++){	
		for(int i=0;i<box_col;i=i+1){
			//printf("row decimal to binary[%d]: %lld\n",x*box_col+i,csvMat[i*2+1+x*box_col*2]);
			decimal_to_binary(csvMat[i*2+1+x*box_col*2],bits_size,binary);
			for(int k=0;k<boxSize;k++){
				int result=0;
				for(int z=0;z<8;z++){//change the binary to the decimal to XOR
   					result=result+binary[z+k*8]*pow(2,7-z);
   					//printf("result is:%d\n",result);
   				}
   				result_matrix[k*box_col+i+boxSize*x*box_col]=result;
   				//printf("checkbox row result %d: result_matrix[%d] = %d\n",x*256+8*i+k,k*box_col+i+boxSize*x*box_col,result_matrix[k*box_col+i+boxSize*x*box_col]);
   			}
		}
	}	
}

void checkbox_binary_column(uint64_t *csvMat,int boxSize,int box_col,int *binary,int *result_matrix,int bits_size){
	for(int x=0;x<box_col;x++){	
		for(int i=0;i<box_col;i=i+1){
			//printf("col decimal to binary[%d]: %lld\n",x*box_col+i,csvMat[i*2+x*box_col*2]);
			decimal_to_binary(csvMat[i*2+x*box_col*2],bits_size,binary);//i*2+x*box_col*2//x*2+i*box_col*2
			for(int k=0;k<boxSize;k++){
				int result=0;
				for(int z=0;z<8;z++){//change the binary to the decimal to XOR
   					result=result+binary[z+k*8]*pow(2,7-z);
   				}
   				result_matrix[i*box_col*boxSize+x+k*box_col]=result;//i*box_col*boxSize+x+k*box_col//k*box_col+i+boxSize*x*box_col
   				//printf("checkbox column result %d: %d\n",x*256+boxSize*i+k,result_matrix[k*box_col+i+boxSize*x*box_col]);
   			}
		}
	}	
}*/

void get_xor(int *result_xor,int *result_matrix,int box_col, int M){
	for(int j=0;j<M;j++){
   		for(int i=0+j*box_col;i<box_col-1+j*box_col;i++){
   	   		result_matrix[i+1]=result_matrix[i]^result_matrix[i+1];//XOR every decimal of the row or column
   	   		//printf("result_matrix %d is %d\n",i+1,result_matrix[i+1]);
   		}
   		result_xor[j] = result_matrix[box_col-1+j*box_col];//get the last one, which is the final result of the XOR
   		//printf("xor of the row or column %d is %d\n",j,result_xor[j]);	
   	}	
}

int *rowXOR(uchar *p, int M){

	int i, j;
	int *row_xor;
	row_xor = (int*) malloc(M*sizeof(int));
	if(row_xor == NULL){ printf("Fail to melloc \n\n"); exit(EXIT_FAILURE); }

	for(i=0;i<M;i++){
		row_xor[i] = p[i*M] ;
	}

	for(i=0;i<M;i++){
		for(j=1;j<M;j++){
			row_xor[i] = row_xor[i] ^ p[i*M+j];
		}
	}
	return row_xor;	
}

int *colXOR(uchar *p, int M){
	int i, j;
	int *col_xor;
	col_xor = (int*) malloc(M*sizeof(int));
	if(col_xor == NULL){ printf("Fail to melloc \n\n"); exit(EXIT_FAILURE); }

	for(i=0;i<M;i++){
		col_xor[i] = p[i] ;
	}

	for(i=0;i<M;i++){
		for(j=1;j<M;j++){
			col_xor[i] = col_xor[i] ^ p[j*M+i];
		}
	}
	return col_xor;
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
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
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

///////////////////unscramble image XOR result////////////////////////////
	row_xor = (int*) malloc(M*sizeof(int));
	if(row_xor == NULL){ printf("Fail to melloc \n\n"); exit(EXIT_FAILURE); }
	col_xor = (int*) malloc(N*sizeof(int));
	if(col_xor == NULL){ printf("Fail to melloc \n\n"); exit(EXIT_FAILURE); }

	uchar *p = image.data;
	row_xor = rowXOR(p, M);
	col_xor = colXOR(p, M);
	//for(i=0; i< M; i++) printf("row_xor[%d] = %d\n", i, row_xor[i]);
	//for(i=0; i< M; i++) printf("col_xor[%d] = %d\n", i, col_xor[i]);

    char buffer[1024] ;
    char *record,*line;
    i = 0;
    j = 0;
    uint64_t csvmat_read[numBox][2];
    uint64_t csvMat[numBox*2];

/////////////////checkbox load/////////////////////////////////////////
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

   //for(int j=0;j<numBox*2;j++){printf("csvmat[%d]:%llu\n",j,csvMat[j]);};

 ///////////////////////////////////////////
////////////some varibles///////////////////
   int bits_size;
   bits_size=boxSize*8;//when boxsize==2 then bits_size=16; when boxsize==4, then bits_size=32; when boxsize==8, then bits_size=64

   int *binary;
   int *result_matrix_row;	
   int *result_matrix_col;
   int *result_xor_row;
   int *result_xor_col;

   binary = (int*) malloc(bits_size*sizeof(int));// this is to store the binary of the box 
   if(binary == NULL){ printf("Fail to melloc binary\n\n"); exit(EXIT_FAILURE); }

   result_matrix_row= (int*) malloc(M*box_col*sizeof(int));// this is to store the decimal which is transformed from the 8 digits
   if(result_matrix_row == NULL){ printf("Fail to melloc result_matrix_row\n\n"); exit(EXIT_FAILURE); }
   //printf("%d\n",M*box_col);
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
   //printf("temp data is %d\n and %d\n",temp.data[0],p[0]);

    struct 		timeval tt;
    double         ST, ET;               // Local Start and num_times for this thread
    long           TE;                   // Local Time Elapsed for this thread
    
    gettimeofday(&tt, NULL);// get the time before the calculation
    ST = tt.tv_sec*1000.00 + (tt.tv_usec/1000.0);
/////////////////////launch the GPU part////////////////////////
  cudaStatus = launch_unscramble(p,csvMat,boxSize,box_col,binary,result_matrix_row,result_matrix_col,bits_size,result_xor_row,result_xor_col,M,GPURuntimes);
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
/////////////////////////////////////////////////////////////////////////////////////////  

  for(int i=0;i<M*box_col;i++){
    printf("%d \n",result_matrix_col[i]);
  }  
/////////////////load checkbox XOR and XOR every line////////////////////////////////////
/////////////////load checkbox for the row, which is the csvmat[][1]/////////////////////
   //checkbox_binary_row(csvMat,boxSize,box_col,binary,result_matrix_row,bits_size);
   get_xor(result_xor_row,result_matrix_row,box_col,M);
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
   	//printf("temp data is %d\n and %d\n",temp.data[0],p[0]);

/////////////////load checkbox XOR and XOR every line////////////////////////////////////
/////////////////load checkbox for the column, which is the csvmat[][1]/////////////////////
    for(int i=0;i<256;i++){
   		swap[i]=0;
    }	

   //checkbox_binary_column(csvMat,boxSize,box_col,binary,result_matrix_col,bits_size);
   get_xor(result_xor_col,result_matrix_col,box_col,M);
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
//////////////////////////////////////
    gettimeofday(&tt, NULL);// get the time after the calculation
    ET = tt.tv_sec*1000.00 + (tt.tv_usec/1000.0);
    TE = (long) (ET-ST); // calculate the total calculating time
    printf(" unscramble the image in %ld ms\n\n", TE); // display the total calculating time
	

///////////////////////////////////////////////////////////////////////////////////////////////////
	// Display the output image:
	Mat result = Mat(M, N, CV_8UC1, image.data);
	// and save it to disk:
	string output_filename = "new.png";
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		exit(EXIT_FAILURE);
	}
	printf("Saved image '%s', size = %dx%d (dims = %d).\n", output_filename.c_str(), result.rows, result.cols, result.dims);

	free(row_xor);
	free(col_xor);
	free(binary);
	free(result_matrix_row);
  free(result_matrix_col);
	free(result_xor_row);
  free(result_xor_col);  
	free(temp_image);
  exit(EXIT_SUCCESS);
}

// Helper function for launching a CUDA kernel (including memcpy, timing, etc.):
cudaError_t launch_unscramble(uchar *p,uint64_t *csvMat,int boxSize,int box_col,int *binary,int *result_matrix_row,int *result_matrix_col,int bits_size,int *result_xor_row,int *result_xor_col,int M,float* Runtimes)
{
  cudaEvent_t time1, time2, time3, time4;
  uint64_t *GPU_csvMat;
  int *GPU_binary;
  int *GPU_result_matrix_row;
  int *GPU_result_matrix_col;
  int *GPU_result_xor_row;
  int *GPU_result_xor_col;

  //dim3 threadsPerBlock;
  //dim3 numBlocks;

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

  cudaStatus = cudaMalloc((void**)&GPU_csvMat, 2*box_col*box_col*sizeof(uint64_t));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_csvMat cudaMalloc failed!\n");
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&GPU_binary, bits_size*sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_binary cudaMalloc failed!\n");
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

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(GPU_csvMat, csvMat, 2*box_col*box_col*sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "GPU_csvMat cudaMemcpy failed!\n");
    goto Error;
  }


  cudaEventRecord(time2, 0);

  // Launch a kernel on the GPU with one thread for each pixel.
  //threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
  //numBlocks = dim3(M / threadsPerBlock.x, N / threadsPerBlock.y);
  unscramble_kernel<<<box_col, box_col, 0>>>(GPU_csvMat,boxSize,box_col,M,GPU_binary,GPU_result_matrix_row,GPU_result_matrix_col,bits_size);
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
 /* cudaStatus = cudaMemcpy(result_xor_row, GPU_result_xor_row,  M*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "result_xor_row cudaMemcpy failed!\n");
    goto Error;
  }
 cudaStatus = cudaMemcpy(result_xor_col, GPU_result_xor_col,  M*sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "result_xor_row cudaMemcpy failed!\n");
    goto Error;
  }  */
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
  cudaFree(GPU_binary);
  cudaFree(GPU_result_matrix_row);
  cudaFree(GPU_result_matrix_col);
  cudaFree(GPU_result_xor_row);
  cudaFree(GPU_result_xor_col);
  cudaEventDestroy(time1);
  cudaEventDestroy(time2);
  cudaEventDestroy(time3);
  cudaEventDestroy(time4);

  return cudaStatus;
}
