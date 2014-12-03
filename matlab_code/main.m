% ECE 406, Fall 2014
% Final Project

% MATLAB version of the assignment, taking an image from unscrambled ->
% scrambled -> unscrambled again.

%% Load image

input_filename = 'unscrambled_image.jpg';
input_image = imread(input_filename);
if size(input_image,3) == 3
    input_image = rgb2gray(input_image);
end

subplot(1,3,1)
imshow(input_image);
title('Original Image')

%% Generate checksums

box_size = 4;  % should be 2, 4, or 8
bpp = 8;  % could read this from the image

% example of how the checksums work:
% a single 2x2 box may be:
%   box = [a b;
%          c d]
% then let:
%   e = a xor c
%   f = b xor d
%   g = a xor b
%   h = c xor d
% the checksums for this box would be:
%   row_checksum = e<<8 + f
%   col_checksum = g<<8 + h

checksums = make_checksums( input_image, box_size, bpp );
dlmwrite( ...
        strcat(input_filename, '_', num2str(box_size),'x',num2str(box_size),'.csv'), ...
        checksums, ...
        'delimiter', ',', ...
        'precision', '%u' ...
        )

%% Scramble image

scramble_level = 1000;

scrambled_image = scramble( input_image, scramble_level );
imwrite(scrambled_image, strcat(input_filename, '_scrambled.jpg'));

subplot(1,3,2)
imshow(scrambled_image);
title('Scrambled Image')

%% Attempt to descramble image

recovered_image = fix_image( scrambled_image, checksums, box_size, bpp );

subplot(1,3,3)
imshow(recovered_image);
title('Recovered Image')
