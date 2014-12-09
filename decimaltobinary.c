 for(int box_index=0;box_index<box_col;box_index++){// for all the box in the image
   		
   		for(int box_col_index=0;box_col_index<boxSize;box_col_index++){// for the one box, which has box
   			int present_row=0;

   			for(int j=0+box_index*64;j<box_col+box_index*64;j++){
   	   			int result=0;
   				decimal_to_binary(csvMat[j+box_index*64][2],bits_size,binary);

   				for(int i=0+8*present_row;i<8+8*present_row;i++){
   					result=result+binary[i-8*present_row]*pow(2,7-i-8*present_row)
   				}
   				result_matrix[j]=result;
   			}

   			for(int i=0;i<box_col-1;i++){
   	   			result_matrix[i+1]=result_matrix[i]^result_matrix[i+1];
   			}
   			result_xor = result_matrix[box_col-1];

   			for(int i=present_row;i<M;i++){//swap from this line
   				if(result_xor==row_xor[i]){// if find the targets, then swap
   					swap_row(present_row,i, p, M);
   					break;
   				}
   			}
   			present_row++;
   		}
   	}