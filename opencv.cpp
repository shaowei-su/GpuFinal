////////////////////////////////////////ECE 406 Final Project///////////////////////////////////////////////////
//                                      CPU Version by Renfei Wang and Shaowei Su                             //
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

///////////////////This function is to calculate the xor of every row in the checkbox/////////////////
void checkbox_binary_row(uint64_t *csvMat,int boxSize,int box_col,int *result_matrix){
  for(int x=0;x<box_col;x++){ 
    for(int i=0;i<box_col;i=i+1){
      uint64_t temp1 = csvMat[i*2+1+x*box_col*2];
      for(int k=0;k<boxSize;k++){
        int result=0;
        uint64_t temp2 = temp1>>8;
        uint64_t temp3 = temp2<<8;
        result = temp1 - temp3;
        temp1 = csvMat[i*2+1+x*box_col*2]>>8*(k+1);
        result_matrix[(boxSize-1-k)*box_col+i+boxSize*x*box_col]=result;
      }
    }
  } 
}
///////////This function is to calculate the xor of every column in the checkbox/////////////////
void checkbox_binary_column(uint64_t *csvMat,int boxSize,int box_col,int *result_matrix){
  for(int x=0;x<box_col;x++){ 
    for(int i=0;i<box_col;i=i+1){
      uint64_t temp1 = csvMat[i*2+x*box_col*2];
      for(int k=0;k<boxSize;k++){
        int result=0;
        uint64_t temp2 = temp1>>8;
        uint64_t temp3 = temp2<<8;
        result = temp1 - temp3;
        temp1 = csvMat[i*2+x*box_col*2]>>8*(k+1);        
        result_matrix[i*box_col*boxSize+x+(boxSize-1-k)*box_col]=result;
        }
    }
  } 
}
//////////This function will get the xor of each row or column///////////////////////
void get_xor(int *result_xor,int *result_matrix,int box_col, int M){
  for(int j=0;j<M;j++){
      for(int i=0+j*box_col;i<box_col-1+j*box_col;i++){
          result_matrix[i+1]=result_matrix[i]^result_matrix[i+1];
      }
      result_xor[j] = result_matrix[box_col-1+j*box_col];//get the last one, which is the final result of the XOR
    } 
}
/////////////This is to calculate the xor of row in the scramble image///////////////
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
/////////////This is to calculate the xor of column in the scramble image///////////////
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
 
   int *result_matrix;  
   int *result_xor;

   result_matrix= (int*) malloc(M*box_col*sizeof(int));// this is to store the decimal which is transformed from the 8 digits
   if(result_matrix == NULL){ printf("Fail to melloc result_matrix\n\n"); exit(EXIT_FAILURE); }
   //printf("%d\n",M*box_col);

   result_xor= (int*) malloc(M*sizeof(int));
   if(result_xor == NULL){ printf("Fail to melloc result_xor\n\n"); exit(EXIT_FAILURE); }

   uchar *temp_image;
   temp_image=(uchar*) malloc(M*N*sizeof(uchar));
   if(temp_image == NULL){ printf("Fail to melloc p\n\n"); exit(EXIT_FAILURE); }

   Mat temp = Mat(M, N, CV_8UC1, temp_image);

    struct    timeval tt;
    double         ST, ET;               // Local Start and num_times for this thread
    double           TE;                   // Local Time Elapsed for this thread
    
    gettimeofday(&tt, NULL);// get the time before the calculation
    ST = tt.tv_sec*1000.00 + (tt.tv_usec/1000.0);

/////////////////load checkbox XOR and XOR every line////////////////////////////////////
/////////////////load checkbox for the row, which is the csvmat[][1]/////////////////////
  row_xor = rowXOR(p, M);
  col_xor = colXOR(p, M);
   checkbox_binary_row(csvMat,boxSize,box_col,result_matrix);
   get_xor(result_xor,result_matrix,box_col,M);

   int flag1=0;
   int flag2=0;
   int swap[256];
   for(int i=0;i<256;i++){
      swap[i]=0;
   }
    for(int j=0;j<N;j++){
      for(int i=0;i<M;i++){//swap from this line
        if(result_xor[j]==row_xor[i] && swap[i]==0){// if find the targets, then swap 
          swap[i]=1;
          Mat M1 = temp.row(j);
          image.row(i).copyTo(M1);
          flag1++;
          //printf("has swaped column %d and %d and the result_xor is %d the row_xor is %d\n",j,i,result_xor[j],row_xor[i]);
          break;
        }
      }
    } 

/////////////////load checkbox XOR and XOR every line////////////////////////////////////
/////////////////load checkbox for the column, which is the csvmat[][1]/////////////////////
    for(int i=0;i<256;i++){
      swap[i]=0;
    } 
   checkbox_binary_column(csvMat,boxSize,box_col,result_matrix);
   get_xor(result_xor,result_matrix,box_col,M);
    for(int j=0;j<N;j++){
      for(int i=0;i<M;i++){//swap from this line
        if(result_xor[j]==col_xor[i] && swap[i]==0){// if find the targets, then swap
          swap[i]=1;
          flag2++;
          Mat M2 = image.col(j);
          temp.col(i).copyTo(M2);
          //printf("has swaped row %d and %dand the result_xor is %d the row_xor is %d\n",j,i,result_xor[j],col_xor[i]);
          break;
        }
      }
    } 
    //printf("%d, %d \n",flag1,flag2);
///////////////////////////////////////////////////////////////////////////////////////
    gettimeofday(&tt, NULL);// get the time after the calculation
    ET = tt.tv_sec*1000.00 + (tt.tv_usec/1000.0);
    TE =  (ET-ST); // calculate the total calculating time
    printf(" unscramble the image in %f ms\n\n", TE); // display the total calculating time


///////////////////////////////////////////////////////////////////////////////////////////////////
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
  free(result_matrix);
  free(result_xor);
  free(temp_image);
  exit(EXIT_SUCCESS);
}