function [ F ] = Wavelet_Main( I )

%https://www.mathworks.com/help/wavelet/ref/dwt2.html#d123e24786
%how to get gabor energy?

gaborFeatures = WaveletGaborFeatures(I);
huurFeatures = WaveletHuurFeatures(I);
mallatFeatures = WaveletMallat(I);
sgaborFeatures = size(gaborFeatures);
shuurFeatures = size(huurFeatures);
smallatFeatures = size(mallatFeatures);
a = max(max(sgaborFeatures(1),shuurFeatures(1)),smallatFeatures(1));
F = [[gaborFeatures;zeros(abs([a, 0]-sgaborFeatures))],[huurFeatures;zeros(abs([a,0]-shuurFeatures))], [mallatFeatures;zeros(abs([a,0]-smallatFeatures))]];
%F = [gaborFeatures, huurFeatures, mallatFeatures];
end