function [n,r] = boxcount(c,varargin)
%BOXCOUNT  Box-Counting of a D-dimensional array (with D=1,2,3).
%   [N, R] = BOXCOUNT(C), where C is a D-dimensional array (with D=1,2,3),
%   counts the number N of D-dimensional boxes of size R needed to cover
%   the nonzero elements of C. The box sizes are powers of two, i.e., 
%   R = 1, 2, 4 ... 2^P, where P is the smallest integer such that
%   MAX(SIZE(C)) <= 2^P. If the sizes of C over each dimension are smaller
%   than 2^P, C is padded with zeros to size 2^P over each dimension (e.g.,
%   a 320-by-200 image is padded to 512-by-512). The output vectors N and R
%   are of size P+1. For a RGB color image (m-by-n-by-3 array), a summation
%   over the 3 RGB planes is done first.
%
%   The Box-counting method is useful to determine fractal properties of a
%   1D segment, a 2D image or a 3D array. If C is a fractal set, with
%   fractal dimension DF < D, then N scales as R^(-DF). DF is known as the
%   Minkowski-Bouligand dimension, or Kolmogorov capacity, or Kolmogorov
%   dimension, or simply box-counting dimension.
%
%   BOXCOUNT(C,'plot') also shows the log-log plot of N as a function of R
%   (if no output argument, this option is selected by default).
%
%   BOXCOUNT(C,'slope') also shows the semi-log plot of the local slope
%   DF = - dlnN/dlnR as a function of R. If DF is contant in a certain
%   range of R, then DF is the fractal dimension of the set C. The
%   derivative is computed as a 2nd order finite difference (see GRADIENT).
%
%   The execution time depends on the sizes of C. It is fastest for powers
%   of two over each dimension.
%
%   Examples:
%
%      % Plots the box-count of a vector containing randomly-distributed
%      % 0 and 1. This set is not fractal: one has N = R^-2 at large R,
%      % and N = cste at small R.
%      c = (rand(1,2048)<0.2);
%      boxcount(c);
%
%      % Plots the box-count and the fractal dimension of a 2D fractal set
%      % of size 512^2 (obtained by RANDCANTOR), with fractal dimension
%      % DF = 2 + log(P) / log(2) = 1.68 (with P=0.8).
%      c = randcantor(0.8, 512, 2);
%      boxcount(c);
%      figure, boxcount(c, 'slope');
%
%   F. Moisy
%   Revision: 2.10,  Date: 2008/07/09


% History:
% 2006/11/22: v2.00, joined into a single file boxcountn (n=1,2,3).
% 2008/07/09: v2.10, minor improvements

% control input argument
error(nargchk(1,2,nargin));

% check for true color image (m-by-n-by-3 array)
if ndims(c)==3
    if size(c,3)==3 && size(c,1)>=8 && size(c,2)>=8
        c = sum(c,3);
    end
end

warning off
c = logical(squeeze(c));
warning on

% transpose the vector to a 1-by-n vector
if length(c)==numel(c)
    dim=1;
    if size(c,1)~=1   
        c = c';
    end   
end

width = max(size(c));    % largest size of the box
p = log(width)/log(2);   % nbre of generations

% remap the array if the sizes are not all equal,
% or if they are not power of two
% (this slows down the computation!)
if p~=round(p) || any(size(c)~=width)
    p = ceil(p);
    width = 2^p;

    mz = zeros(width, width);
    mz(1:size(c,1), 1:size(c,2)) = c;
    c = mz;

end

n=zeros(1,p+1); % pre-allocate the number of box of size r



n(p+1) = sum(c(:));
for g=(p-1):-1:0
    siz = 2^(p-g);
    siz2 = round(siz/2);
    for i=1:siz:(width-siz+1)
        for j=1:siz:(width-siz+1)
            c(i,j) = ( c(i,j) || c(i+siz2,j) || c(i,j+siz2) || c(i+siz2,j+siz2) );
        end
    end
    n(g+1) = sum(sum(c(1:siz:(width-siz+1),1:siz:(width-siz+1))));
        
end
    


n = n(end:-1:1);
r = 2.^(0:p); % box size (1, 2, 4, 8...)

% if any(strncmpi(varargin,'slope',1))
%     s=-gradient(log(n))./gradient(log(r));
%     semilogx(r, s, 's-');
%     ylim([0 2]);
%     xlabel('r, box size'); ylabel('- d ln n / d ln r, local dimension');
%     title([num2str(2) 'D box-count']);
% elseif nargout==0 || any(strncmpi(varargin,'plot',1))
%     loglog(r,n,'s-');
%     xlabel('r, box size'); ylabel('n(r), number of boxes');
%     title([num2str(2) 'D box-count']);
% end
if nargout==0
    clear r n
end
