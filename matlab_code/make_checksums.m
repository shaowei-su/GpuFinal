function [ checksums ] = make_checksums( data, box_size, bpp )
%MAKE_CHECKSUMS Return row and column checksums for each box in data.
%   data is the full matrix (e.g. read in by imread(some_file.png))
%   box_size is the width/height of a 'box' in data that will be checksummed
%   bpp is bits per pixel, usually 8 for grayscale

% checksum_width_in_bits = box_size*bpp

num_boxes = size(data,1)*size(data,2) / box_size^2;
boxes_per_row = size(data,2) / box_size;

checksums = uint64(zeros(num_boxes,2));  % row = box number.  col1/2 = row/col checksum.

for box_num = 1:num_boxes
    
    % these offsets are 0 indexed:
    offset_row = floor((box_num-1) / boxes_per_row)*box_size;
    offset_col = mod(box_num-1, boxes_per_row)*box_size;

    box = data(offset_row+1:offset_row+box_size, ...
               offset_col+1:offset_col+box_size);
    
    checksums(box_num, 1) = row_checksum( box , bpp );
    checksums(box_num, 2) = row_checksum( box', bpp );
    
end

end

function [ checksum ] = row_checksum( box, bpp )

box = uint64(box);

col_checksums = uint64(zeros(1,size(box,2)));
for i = 1:size(box,1)  % for each row in this box
    % xor with the previous rows
    col_checksums = bitxor(col_checksums, box(i,:));
end

% col_checksums is still split into separate columns.  next we'll
% concatenate them.

checksum = uint64(0);
for i = 1:size(box,2)  % for each column (e.g. byte) of the checksum
    checksum = checksum + bitshift( col_checksums(i), bpp*(size(box,2)-i) );
end

end
