function [ scrambled_image ] = scramble( original_data, scrambling_level )
%SCRAMBLE Randomly swap rows and columns of original_image.
%   original_image is the matrix from e.g. imread(filename).
%   scrambling_level is the number of swaps to make.
%   The scrambled image is returned in the same form as imread()'s output.

% Note that we could use randperm() to generate the swaps, but then we
% wouldn't be able to set the 'scrambling level'.

scrambled_image = original_data;

for i = 1:scrambling_level
    
    % coin toss to see if we're working on rows or columns, weighted based
    % on image dimensions.  1 means swap two rows, 2 means swap two columns.
    swap_dimension = randsample([1,2],1,true,[size(original_data,1),size(original_data,2)]);

    % pick 2 different random numbers in range of rows or columns
    first_thing = randi(size(original_data,swap_dimension));
    second_thing = first_thing;
    while (second_thing == first_thing)
        second_thing = randi(size(original_data,swap_dimension));
    end
    
    % swap those two rows or columns
    if (swap_dimension == 1)
        temp = scrambled_image(first_thing,:);
        scrambled_image(first_thing,:) = scrambled_image(second_thing,:);
        scrambled_image(second_thing,:) = temp;
%         fprintf('swapped row %d and %d\n', first_thing, second_thing)  % debugging
    else
        temp = scrambled_image(:,first_thing);
        scrambled_image(:,first_thing) = scrambled_image(:,second_thing);
        scrambled_image(:,second_thing) = temp;
%         fprintf('swapped column %d and %d\n', first_thing, second_thing)  % debugging
    end
    
end

end