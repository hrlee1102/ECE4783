path_directory = 'Image_data'; 
original_files = dir([path_directory '/*.png']);

target_necrosis = rgb2lab(imread("Image_data/Necrosis_1.png")); % Target image for Necrosis
target_stroma = rgb2lab(imread("Image_data/Stroma_1.png")); % Target image for Stroma
target_tumor = rgb2lab(imread("Image_data/Tumor_1.png")); % Target image for Tumor
featureN = []; 
featureS = []; 
featureT = []; 
total = [];
%featureFractal = fractal_Main(A);
%featureGLCM = GLCM(A);
%featurerWavelet = Wavelet_Main(A);
for k=2:100 % Skip the first image since it's target
    filenameN = [path_directory '/' original_files(k).name]; % Get image name for Necrosis Image
    filenameS = [path_directory '/' original_files(k+100).name]; % Get image name for Stroma Image
    filenameT = [path_directory '/' original_files(k+200).name]; % Get image name for Tumor Image
    featureN = [featureN, fractal_Main(imread(filenameN)), GLCM(imread(filenameN)), Wavelet_Main(imread(filenameN))];
    featureS = [featureS, fractal_Main(imread(filenameS)), GLCM(imread(filenameS)), Wavelet_Main(imread(filenameS))];
    featureT = [featureT, fractal_Main(imread(filenameT)), GLCM(imread(filenameT)), Wavelet_Main(imread(filenameT))];

end

total = [featureN, featureS, featureT];
save("featureTexturer.mat", "total");
