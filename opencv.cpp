
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <stdlib.h>
#include <math.h>

using namespace cv;


int M;  // number of rows in image
int N;  // number of columns in image
int numBox;
int boxSize;
int box_col; // equals to the box_row

//decimal to binary function
//the decimal is the number that would to tranform
//bits_size is how many binary bits that would to transform
//*binary is the array to store the binary bits
void decimal_to_binary(int decimal, int bits_size, int *binary){
	int digit;
	int n=bits_size-1;
	  for (bits_size=n; bits_size >= 0; bits_size--)
	  {
	    digit = decimal >> bits_size;// csv_number is the number
	 
	    if (digit & 1){

		  binary[n-bits_size]=1;
		}

	    else{

		  binary[n-bits_size]=0;
		}
	  }			
}
////////////////////////////////
/*int binary_to_decimal(int bits_size,int *binary){
	int i=0;
	int result=0;
	for(i;i<8;i++){
		result=result+binary[i]*pow(2,i);
	}
	return result;
}*/

void swap_row(int present_row, int target_row,uchar *p, int M){
	int i;
	int temp;
	for(i=0;i<M;i++){
		temp=p[present_row*M+i];
		p[present_row*M+i]=p[target_row*M+i];
		p[target_row*M+i]=temp;
	}	
}	

void swap_column(int present_column, int target_column,uchar *p, int N){
	int i;
	int temp;
	for(i=0;i<N;i++){
		temp=p[present_column+i*N];
		p[present_column+i*N]=p[target_column+i*N];
		p[present_column+i*N]=temp;
	}	
}	

void checkbox_binary_row(long long *csvMat,int boxSize,int box_col,int *binary,long long *result_matrix,int bits_size){
	for(int x=0;x<box_col;x++){	
		for(int i=0;i<box_col;i=i+1){
			decimal_to_binary(csvMat[i*2+1+x*box_col*2],bits_size,binary);
			for(int k=0;k<boxSize;k++){
				int result=0;
				for(int z=0;z<8;z++){//change the binary to the decimal to XOR
   					result=result+binary[z+k*8]*pow(2,7-z);
   				}
   				result_matrix[k*box_col+i+boxSize*x*box_col]=result;
   				//printf("checkbox row result%d: %lld\n",x*256+8*i+k,result_matrix[k*box_col+i+4*x*box_col]);
   			}
		}
	}	
}

void checkbox_binary_column(long long *csvMat,int boxSize,int box_col,int *binary,long long *result_matrix,int bits_size){
	for(int x=0;x<box_col;x++){	
		for(int i=0;i<box_col;i=i+1){
			decimal_to_binary(csvMat[x*2+i*box_col*2],bits_size,binary);
			for(int k=0;k<boxSize;k++){
				int result=0;
				for(int z=0;z<8;z++){//change the binary to the decimal to XOR
   					result=result+binary[z+k*8]*pow(2,7-z);
   				}
   				result_matrix[k*box_col+i+boxSize*x*box_col]=result;
   				//printf("checkbox column result %d: %lld\n",x*256+8*i+k,result_matrix[k*box_col+i+4*x*box_col]);
   			}
		}
	}	
}

void get_xor(long long *result_xor,long long *result_matrix,int box_col, int M){
	for(int j=0;j<M;j++){
   		for(int i=0+j*box_col;i<box_col-1+j*box_col;i++){
   	   		result_matrix[i+1]=result_matrix[i]^result_matrix[i+1];//XOR every decial of the row or column
   	   		//printf("result_matrix %d is %lld\n",i,result_matrix[i+1]);
   		}
   		result_xor[j] = result_matrix[box_col-1+j*box_col];//get the last one, which is the final result of the XOR
   		//printf("xor of the row or column %d is %lld\n",j,result_xor[j]);	
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
	for(i=0; i< M; i++) printf("row_xor[%d] = %d\n", i, row_xor[i]);
	//for(i=0; i< M; i++) printf("col_xor[%d] = %d\n", i, col_xor[i]);

    char buffer[1024] ;
    char *record,*line;
    i = 0;
    j = 0;
    long long csvMat[numBox*2];

/////////////////checkbox load/////////////////////////////////////////
    FILE *fstream = fopen(argv[2],"r");
    if(fstream == NULL)
    {
       	printf("\n file opening failed ");
       	exit(EXIT_FAILURE);
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
    {	
    	
      	record = strtok(line,",");
      	while(record != NULL)
      	{ 
      		csvMat[i] = atoll(record) ;
      		//printf("record : %lld at %d, %d \n", csvMat[i][j], i, j) ; 
      		record = strtok(NULL,",");
      		j++ ;
      	}
      	++i ;
   }

 ///////////////////////////////////////////
////////////some varibles///////////////////
   int bits_size;
   bits_size=boxSize*8;
   /* this is the comment of the above line
   if(boxSize==2){
   		bits_size=16;
   }
   else if(boxSize==4){
   		bits_size=32;
   }
   else if(boxSize==8){
   		bits_size=64;
   }*/
   int *binary;
   long long*result_matrix;	
   long long*result_xor;

   binary = (int*) malloc(bits_size*sizeof(int));// this is to store the binary of the box 
   if(binary == NULL){ printf("Fail to melloc binary\n\n"); exit(EXIT_FAILURE); }

   result_matrix= (long long*) malloc(M*box_col*sizeof(long long));// this is to store the decimal which is transformed from the 8 digits
   if(result_matrix == NULL){ printf("Fail to melloc result_matrix\n\n"); exit(EXIT_FAILURE); }

   result_xor= (long long*) malloc(M*sizeof(long long));
   if(result_xor == NULL){ printf("Fail to melloc result_xor\n\n"); exit(EXIT_FAILURE); }


/////////////////load checkbox XOR and XOR every line////////////////////////////////////
/////////////////load checkbox for the row, which is the csvmat[][1]/////////////////////
   checkbox_binary_row(csvMat,boxSize,box_col,binary,result_matrix,bits_size);
   get_xor(result_xor,result_matrix,box_col,M);
   	for(int j=0;j<N;j++){
   		for(int i=j;i<M;i++){//swap from this line
   			if(result_xor[j]==row_xor[i]){// if find the targets, then swap
   				swap_row(j,i, p, M);
   				//printf("has swaped column %d and %d and the result_xor is %lld the row_xor is %d\n",j,i,result_xor[j],row_xor[i]);
   				break;
   			}
   		}
   	}	
/////////////////load checkbox XOR and XOR every line////////////////////////////////////
/////////////////load checkbox for the column, which is the csvmat[][1]/////////////////////
   checkbox_binary_column(csvMat,boxSize,box_col,binary,result_matrix,bits_size);
   get_xor(result_xor,result_matrix,box_col,M);
   	for(int j=0;j<N;j++){
   		for(int i=j;i<M;i++){//swap from this line
   			if(result_xor[j]==col_xor[i]){// if find the targets, then swap
   				swap_column(j,i, p, M);
   				//printf("has swaped row %d and %dand the result_xor is %lld the row_xor is %d\n",j,i,result_xor[j],col_xor[i]);
   				break;
   			}
   		}
   	}	
	

////////////////////////////
   	row_xor = rowXOR(p, M);
	col_xor = colXOR(p, M);
	for(i=0; i< M; i++) printf("row_xor[%d] = %d\n", i, row_xor[i]);
	//for(i=0; i< M; i++) printf("col_xor[%d] = %d\n", i, col_xor[i]);
   	image.data=p; // output the new image data
	// Display the output image:
	Mat result = Mat(M, N, CV_8UC1, p);
	// and save it to disk:
	string output_filename = "new.png";
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		exit(EXIT_FAILURE);
	}
	printf("Saved image '%s', size = %dx%d (dims = %d).\n", output_filename.c_str(), result.rows, result.cols, result.dims);

	free(row_xor);
	free(col_xor);
	return 1;
}