% This generates the Cartesian product of elements from input cell arrays.
% It takes any number of cell arrays as input and returns a cell array
% containing all possible combinations of the elements from the input.
% Each row of the output corresponds to one unique combination.

function output = product_of_cells(varargin)
    % Process input
    input = varargin(1:nargin);
    L = length(input);
    assert(all(cellfun(@iscell, input)), "We need everything to be a cell");

    % ndgrid generates L grids of indices, one for each input
    [ix{1:L}] = ndgrid(input{:});

    % Pre-allocate the output where the number of rows is the total
    % number of combinations (numel(ix{1})) and the number of columns
    % is equal to the number of input cell arrays (L)
    output = cell(numel(ix{1}), L);

    for k = 1:L
        output(:, k) = reshape(ix{k}, [], 1);
    end
end
