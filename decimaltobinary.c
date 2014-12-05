#include <stdio.h>

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
 
int main()
{
  int csv_number, bits_size, digit;
  int binary[32];
  
  decimal_to_binary(5,32,binary);	

	for(int i=1;i<=32;i++){
		printf("%d",binary[i-1]);
	}
 
  return 0;
}