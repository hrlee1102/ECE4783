function [ F ] = fractal_Main( I )
%This function combine all fractal feature into 1-d vector
%https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/13063/versions/1/previews/boxcount/html/demo.html
%https://www.mathworks.com/matlabcentral/fileexchange/25261-lacunarity-of-a-binary-image?s_tid=srchtitle


A = im2bw(I);

[n,r] = fractalBoxcount(A, 'slope');

%dimension
[ D ] = fractalHausDim( A );
%box
box = permute(n, [2,1]);
% %Lacunarity
% [rows, cols] = size(A);
% A = 1 - A;
% %%
% n = 2;
% while(n <= rows)
% nn = n-1;
% rnn = rows - nn;
% index = uint8(log2(n));
% count(index)= power(rnn,2);
% sigma(index) = 0.0;
% sigma2(index) = 0.0;
% for i=1:rnn
%     for j=1:rnn
%         sums = sum(sum(A(i:i+nn,j:j+nn)));
%         sigma(index) = sigma(index) + sums;
%         sigma2(index) = sigma2(index) + power(sums,2);
%     end
% end
% n = n * 2;
% end
% %%
% for i=1:index
%     
%     M(i,1)= (count(i)*sigma2(i))/(power(sigma(i),2));
% end
%M = permute(M, [2 1]);
sD = size(D);
sbox = size(box);
a = max(sD(1),sbox(1));
F = [[D;zeros(abs([a 0]-sD))],[box;zeros(abs([a,0]-sbox))]];
end