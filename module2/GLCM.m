function [ F ] = GLCM( I )
A = rgb2gray(I);

%glcms of 64*64 matrix, distance 1-5
glcms = graycomatrix(A, 'NumLevel', 64, 'Offset', [0 1; 0 2; 0 3; 0 4; 0 5]);
%glcmInertia, glcmEnergy, glcmCorrelation, glcmHomogeneity
stats = graycoprops(glcms, {'contrast', 'Energy', 'Correlation', 'Homogeneity'});
%glcmEntropy
glcmEntropy = zeros(1,5);
gProb = zeros(64);
%glcmMaxPro
glcmMaxProb = zeros(1,5);
%CLCMClusterShade
glcmClusterShadeMx = zeros(size(glcms));
glcmClusterShadeMy = zeros(size(glcms));
glcmClusterShade = zeros(size(glcms));
%CLCMClusterShade
glcmClusterProminenceMx = zeros(1,5);
glcmClusterProminenceMy = zeros(1,5);
glcmClusterProminence = zeros(size(glcms));


for k= 1:5
    gProb = glcms(:,:,k)./sum(glcms(:,:,k));
    gProb(isnan(gProb))=0;
    glcmMaxProb(k) = max(gProb, [],'all');
    glcmEntropy(k) = entropy(glcms(:,:,k));
    for i= 1:64
        for j = 1:64 
            glcmClusterShadeMx(i,j,k) = glcms(i,j,k)*i;
            glcmClusterShadeMy(i,j,k) = glcms(i,j,k)*i;
        end
    end
end

glcmClusterShadeMx = reshape(sum(sum(glcmClusterShadeMx)), 1, 5);
glcmClusterShadeMy = reshape(sum(sum(glcmClusterShadeMy)), 1, 5);
glcmClusterProminenceMx = glcmClusterShadeMx;
glcmClusterProminenceMy = glcmClusterShadeMy;

for k= 1:5
    for i= 1:64
        for j = 1:64
            
            glcmClusterShade(i,j,k) = (k-glcmClusterShadeMx(k)+j-glcmClusterShadeMy(k))^3*glcms(i,j,k);
            glcmClusterProminence(i,j,k) = (k-glcmClusterShadeMx(k)+j-glcmClusterShadeMy(k))^4*glcms(i,j,k);
        end
    end
end

glcmClusterShade = reshape(sum(sum(glcmClusterShade)), 1, 5);
glcmClusterProminence = reshape(sum(sum(glcmClusterProminence)), 1, 5);

stats.Entropy = glcmEntropy; %save glcmEntropy to stats structure
stats.MaxProb = glcmMaxProb; %save glcmMaxProb to stats structure
stats.ClusterShade = glcmClusterShade;
stats.ClusterProminence = glcmClusterProminence;

F = [stats.Contrast, stats.Energy, stats.Correlation, stats.Homogeneity, glcmEntropy, glcmMaxProb, glcmClusterShade, glcmClusterProminence];
end