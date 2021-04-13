function [ F ] = Wavelet_Main( I )

%https://www.mathworks.com/help/wavelet/ref/dwt2.html#d123e24786
%how to get gabor energy?

gaborFeatures = WaveletGaborFeatures(I);
huurFeatures = WaveletHuurFeatures(I);
mallatFeatures = WaveletMallat(I);
F = [gaborFeatures, huurFeatures, mallatFeatures];
end