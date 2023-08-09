import numpy as np
import warnings


def allcomb(varargin):
    # ALLCOMB - All combinations
    #    B = ALLCOMB(A1,A2,A3,...,AN) returns all combinations of the elements
    #    in the arrays A1, A2, ..., and AN. B is P-by-N matrix where P is the product
    #    of the number of elements of the N inputs.
    #    This functionality is also known as the Cartesian Product. The
    #    arguments can be numerical and/or characters, or they can be cell arrays.

    #    Examples:
    #       allcomb([1 3 5],[-3 8],[0 1]) # numerical input:
    #       # -> [ 1  -3   0
    #       #      1  -3   1
    #       #      1   8   0
    #       #        ...
    #       #      5  -3   1
    #       #      5   8   1 ] ; # a 12-by-3 array

    #       allcomb('abc','XY') # character arrays
    #       # -> [ aX ; aY ; bX ; bY ; cX ; cY] # a 6-by-2 character array

    #       allcomb('xy',[65 66]) # a combination -> character output
    #       # -> ['xA' ; 'xB' ; 'yA' ; 'yB'] # a 4-by-2 character array

    #       allcomb({'hello','Bye'},{'Joe', 10:12},{99999 []}) # all cell arrays
    #       # -> {  'hello'  'Joe'        [99999]
    #       #       'hello'  'Joe'             []
    #       #       'hello'  [1x3 double] [99999]
    #       #       'hello'  [1x3 double]      []
    #       #       'Bye'    'Joe'        [99999]
    #       #       'Bye'    'Joe'             []
    #       #       'Bye'    [1x3 double] [99999]
    #       #       'Bye'    [1x3 double]      [] } ; # a 8-by-3 cell array

    #    ALLCOMB(..., 'matlab') causes the first column to change fastest which
    #    is consistent with matlab indexing. Example:
    #      allcomb(1:2,3:4,5:6,'matlab')
    #      # -> [ 1 3 5 ; 1 4 5 ; 1 3 6 ; ... ; 2 4 6 ]

    #    If one of the N arguments is empty, ALLCOMB returns a 0-by-N empty array.

    #    See also NCHOOSEK, PERMS, NDGRID
    #         and NCHOOSE, COMBN, KTHCOMBN (Matlab Central FEX)

    # Tested in Matlab R2015a and up
    # version 4.2 (apr 2018)
    # (c) Jos van der Geest
    # email: samelinoa@gmail.com

    # History
    # 1.1 (feb 2006), removed minor bug when entering empty cell arrays;
    #     added option to let the first input run fastest (suggestion by JD)
    # 1.2 (jan 2010), using ii as an index on the left-hand for the multiple
    #     output by NDGRID. Thanks to Jan Simon, for showing this little trick
    # 2.0 (dec 2010). Bruno Luong convinced me that an empty input should
    # return an empty output.
    # 2.1 (feb 2011). A cell as input argument caused the check on the last
    #      argument (specifying the order) to crash.
    # 2.2 (jan 2012). removed a superfluous line of code (ischar(..))
    # 3.0 (may 2012) removed check for doubles so character arrays are accepted
    # 4.0 (feb 2014) added support for cell arrays
    # 4.1 (feb 2016) fixed error for cell array input with last argument being
    #     'matlab'. Thanks to Richard for pointing this out.
    # 4.2 (apr 2018) fixed some grammar mistakes in the help and comments

    narginchk(1, Inf)
    NC = len(varargin)
    # check if we should flip the order
    if ischar(varargin[end()]) and (strcmpi(varargin[end()], "matlab") or strcmpi(varargin[end()], "john")):
        # based on a suggestion by JD on the FEX
        NC = NC - 1
        ii = np.arange(1, NC + 1)
    else:
        # default: enter arguments backwards, so last one (AN) is changing fastest
        ii = np.arange(NC, 1 + -1, -1)

    args = varargin(np.arange(1, NC + 1))
    if np.any(cellfun("isempty", args)):
        warnings.warn("ALLCOMB:EmptyInput", "One of more empty inputs result in an empty output.")
        A = np.zeros((0, NC))
    else:
        if NC == 0:
            A = np.zeros((0, 0))
        else:
            if NC == 1:
                A = args[0]
            else:
                isCellInput = cellfun(iscell, args)
                if np.any(isCellInput):
                    if not np.all(isCellInput):
                        raise Exception("ALLCOMB:InvalidCellInput", "For cell input, all arguments should be cell arrays.")
                    # for cell input, we use to indices to get all combinations
                    ix = cellfun(lambda c: np.arange(1, np.asarray(c).size + 1), args, "un", 0)
                    # flip using ii if last column is changing fastest
                    ix[ii] = ndgrid(ix[ii])
                    A = cell(np.asarray(ix[0]).size, NC)
                    for k in np.arange(1, NC + 1).reshape(-1):
                        # combine
                        A[:, k] = reshape(args[k](ix[k]), [], 1)
                else:
                    # non-cell input, assuming all numerical values or strings
                    # flip using ii if last column is changing fastest
                    A[ii] = ndgrid(args[ii])
                    # concatenate
                    A = reshape(cat(NC + 1, A[:]), [], NC)

    return A
