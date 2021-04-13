function [F] = WaveletHuurFeatures(I)

%https://www.mathworks.com/help/wavelet/ref/dwt2.html#d123e24786
A = imread('/Users/kyeomeunjang/Desktop/ECE_4783/img/Necrosis_1.png');
Agray = rgb2gray(A);
[LoD,HiD] = wfilters('haar','d');
[cA,cH,cV,cD] = dwt2(Agray,LoD,HiD,'mode','symh');

%huur_entropy
huurCAEntropy = entropy(cA);
huurCHEntropy = entropy(cH);
huurCVEntropy = entropy(cV);
huurCDEntropy = entropy(cD);

%Garbor_Energy
huurCAEnergy = sum(cA(:));
huurCHEnergy = sum(cH(:));
huurCVEnergy = sum(cV(:));
huurCDEnergy = sum(cD(:));

F = [huurCAEntropy, huurCHEntropy, huurCVEntropy, huurCDEntropy, huurCAEnergy, huurCHEnergy, huurCVEnergy, huurCDEnergy];
end
