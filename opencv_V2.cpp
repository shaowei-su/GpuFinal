
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
	for(i;i<bits_size;i++){
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
		temp=p[present_column*N+i];
		p[present_column*N+i]=p[target_column*M+i];
		p[present_column*N+i]=temp;
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
    long long csvMat[numBox][2];

    FILE *fstream = fopen(argv[2],"r");
    if(fstream == NULL)
    {
       	printf("\n file opening failed ");
       	exit(EXIT_FAILURE);
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
    {	
    	j= 0;
      	record = strtok(line,",");
      	while(record != NULL)
      	{ 
      		csvMat[i][j] = atoll(record) ;
      		//printf("record : %lld at %d, %d \n", csvMat[i][j], i, j) ; 
      		record = strtok(NULL,",");
      		j++ ;
      	}
      	++i ;
   }

 ///////////////////////////////////////////
//this is the transform of the decimal number to binary
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
   int *result_matrix;	

   binary = (int*) malloc(bits_size*sizeof(int));// this is to store the binary of the box 
   if(binary == NULL){ printf("Fail to melloc binary\n\n"); exit(EXIT_FAILURE); }

   result_matrix= (int*) malloc(box_col*sizeof(int));// this is to store the decimal which is transformed from the 8 digits
   if(result_matrix == NULL){ printf("Fail to melloc result_matrix\n\n"); exit(EXIT_FAILURE); }

   int result_xor;// this is to store the final xor of each column and row

/////////////////////////////////////////////////////////////////////////////////////////
/////////////this is the re-swap of the row//////////////////////////////////////////////   
    for(int box_index=0;box_index<box_col;box_index++){// for all the box in the image
   		
   		for(int present_row=0;present_row<boxSize;present_row++){// for the one box, which has bits_size rows and cols

   			for(int j=0+box_index*box_col;j<box_col+box_index*box_col;j++){// for every box to calculate the binary and then decimal
   	   			int result=0;
   				decimal_to_binary(csvMat[j][1],bits_size,binary);/////may has some problems this line

   				for(int i=0+8*present_row;i<8+8*present_row;i++){//change the binary to the decimal to XOR
   					result=result+binary[i]*pow(2,7-i-8*present_row);
   				}
   				result_matrix[j]=result;//store every decimal into the matrix
   			}

   			for(int i=0;i<box_col-1;i++){
   	   			result_matrix[i+1]=result_matrix[i]^result_matrix[i+1];//XOR every decial of the row or column
   			}
   			result_xor = result_matrix[box_col-1];//get the last one, which is the final result of the XOR

   			for(int i=present_row;i<M;i++){//swap from this line
   				if(result_xor==row_xor[i]){// if find the targets, then swap
   					swap_row(present_row,i, p, M);
   					break;
   				}
   			}
   		}
   	}
/////////////////////////////////////////////////////////////////////////////////////////
/////////////this is the re-swap of the column/////////////////////////////////////////// 
     for(int box_index=0;box_index<box_col;box_index++){// for all the box in the image
   		
   		for(int present_column=0;present_column<boxSize;present_column++){// for the one box, which has bits_size rows and cols

   			for(int j=0+box_index*box_col;j<box_col+box_index*box_col;j++){// for every box to calculate the binary and then decimal
   	   			int result=0;
   				decimal_to_binary(csvMat[j*box_col+box_index][0],bits_size,binary);

   				for(int i=0+8*present_column;i<8+8*present_column;i++){//change the binary to the decimal to XOR
   					result=result+binary[i]*pow(2,7-i-8*present_column);
   				}
   				result_matrix[j]=result;//store every decimal into the matrix
   			}

   			for(int i=0;i<box_col-1;i++){
   	   			result_matrix[i+1]=result_matrix[i]^result_matrix[i+1];//XOR every decial of the row or column
   			}
   			result_xor = result_matrix[box_col-1];//get the last one, which is the final result of the XOR

   			for(int i=present_column;i<N;i++){//swap from this line
   				if(result_xor==col_xor[i]){// if find the targets, then swap
   					swap_column(present_column,i, p, N);
   					break;
   				}
   			}
   		}
   	}   

////////////////////////////

   	image.data=p; // output the new image data
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
	printf("sb\n");
	return 1;
}