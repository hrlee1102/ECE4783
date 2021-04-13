function [F] = WaveletGaborFeatures(I)
% 28 magnitude features
%https://www.mathworks.com/help/images/texture-segmentation-using-gabor-filters.html#TextureSegmentationUsingGaborFiltersExample-2
%I = imread('/Users/kyeomeunjang/Desktop/ECE_4783/img/Necrosis_1.png'); 
Agray = rgb2gray(I);

imageSize = size(Agray);
numRows = imageSize(1);
numCols = imageSize(2);

wavelengthMin = 4/sqrt(2);
wavelengthMax = hypot(numRows,numCols);
n = floor(log2(wavelengthMax/wavelengthMin));
wavelength = 2.^(0:(n-2)) * wavelengthMin;

deltaTheta = 45;
orientation = 0:deltaTheta:(180-deltaTheta);

g = gabor(wavelength,orientation);
%Gabor_mag
gaborMag = imgaborfilt(Agray,g);

%Gabor_entropy
%gaborEntropy = zeros(1,28);

% for i = 1:28
%   gaborEntropy(i) = entropy(gaborMag(:,:,i));
% 
% end

%reshape gaborMag
% gaborMag = permute(gaborMag, [2, 1, 3]);
% gaborMag = permute(gaborMag(:), [2, 1, 3]);

gaborMag = permute(gaborMag, [2, 1, 3]);
gaborMag = reshape(permute(gaborMag(:), [2, 1, 3]), [512*512, 28]);
% sgaborMag = size(gaborMag);
% sgaborEntropy = size(gaborEntropy);
%a = max(sgaborMag(1),sgaborEntropy(1));
%z = [[gaborMag;zeros(abs([a 0]-sgaborMag))],[gaborEntropy;zeros(abs([a,0]-sgaborEntropy))]];
%F = [gaborMag, gaborEntropy];
%end
%c = {gaborMag, gaborEntropy};
F = gaborMag;
end