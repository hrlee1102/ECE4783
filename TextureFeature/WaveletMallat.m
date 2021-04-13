function [F] = WaveletMallat(I)

%https://www.mathworks.com/help/wavelet/ref/dwt2.html#d123e24786
%how to get gabor energy?
Agray = rgb2gray(I);

[cA,cH,cV,cD] = dwt2(Agray,'sym4','mode','per');


%Mallat_entropy
mallatCAEntropy = entropy(cA);
mallatCHEntropy = entropy(cH);
mallatCVEntropy = entropy(cV);
mallatCDEntropy = entropy(cD);

%Mallat_Energy
mallatCAEnergy = sum(cA(:));
mallatCHEnergy = sum(cH(:));
mallatCVEnergy = sum(cV(:));
mallatCDEnergy = sum(cD(:));

F = [mallatCAEntropy, mallatCHEntropy, mallatCVEntropy, mallatCDEntropy, mallatCAEnergy, mallatCHEnergy, mallatCVEnergy, mallatCDEnergy];
end