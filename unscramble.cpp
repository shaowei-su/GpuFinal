
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


int main(int argc, char *argv[]){


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

	printf("here the image[0] = %d\n", image.data[0] );

    char buffer[1024] ;
    char *record,*line;
    int i=0,j=0;
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
      		printf("record : %lld at %d, %d \n", csvMat[i][j], i, j) ; 
      		record = strtok(NULL,",");
      		j++ ;
      	}
      	++i ;
   }
//////////////


//////////////
	// Display the output image:
	Mat result = Mat(M, N, CV_8UC1, image.data);
	// and save it to disk:
	string output_filename = "new.png";
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		exit(EXIT_FAILURE);
	}
	printf("Saved image '%s', size = %dx%d (dims = %d).\n", output_filename.c_str(), result.rows, result.cols, result.dims);

	return 1;
}