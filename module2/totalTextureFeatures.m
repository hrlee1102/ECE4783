path_directoryN = '/Users/kyeomeunjang/Desktop/ECE4783/Image_data_normalized/Necrosis'; 
path_directoryS = '/Users/kyeomeunjang/Desktop/ECE4783/Image_data_normalized/Stroma'; 
path_directoryT = '/Users/kyeomeunjang/Desktop/ECE4783/Image_data_normalized/Tumor'; 

original_filesN = dir([path_directoryN '/*.png']);
original_filesS = dir([path_directoryS '/*.png']);
original_filesT = dir([path_directoryT '/*.png']);


featureN = []; 
featureS = []; 
featureT = []; 
colorFeaturesN = [];
colorFeaturesS = [];
colorFeaturesT = [];
MorphoFeaturesN = [];
MorphoFeaturesS = [];
MorphoFeaturesT = [];
total = [];
%featureFractal = fractal_Main(A);
%featureGLCM = GLCM(A);
%featurerWavelet = Wavelet_Main(A);

for k=2:2 % Skip the first image since it's target
    filenameN = [path_directoryN '/' original_filesN(k).name]; % Get image name for Necrosis Image
    filenameS = [path_directoryS '/' original_filesS(k).name]; % Get image name for Stroma Image
    filenameT = [path_directoryT '/' original_filesT(k).name]; % Get image name for Tumor Image
         fracN = fractal_Main(imread(filenameN));
         glcmN = GLCM(imread(filenameN));
         waveN = Wavelet_Main(imread(filenameN));
         fracS = fractal_Main(imread(filenameS));
         glcmS = GLCM(imread(filenameS));
         waveS = Wavelet_Main(imread(filenameS));
         fracT = fractal_Main(imread(filenameT));
         glcmT = GLCM(imread(filenameT));
         waveT = Wavelet_Main(imread(filenameT));
         sfrac = size(fracN);
         sglcm = size(glcmN);
         swave = size(waveN);
         a = max(max(sfrac(1),sglcm(1)),swave(1));
        featureN = [featureN;[fracN;zeros(abs([a, 0]-sfrac))],[glcmN;zeros(abs([a,0]-sglcm))], [waveN;zeros(abs([a,0]-swave))]];
        featureS = [featureS; [fracS;zeros(abs([a, 0]-sfrac))],[glcmS;zeros(abs([a,0]-sglcm))], [waveS;zeros(abs([a,0]-swave))]];
        featureT = [featureT; [fracT;zeros(abs([a, 0]-sfrac))],[glcmT;zeros(abs([a,0]-sglcm))], [waveT;zeros(abs([a,0]-swave))]];
      colorFeaturesN = extract_colorFeatures(imread(filenameN));
      colorFeaturesS = [colorFeaturesS; extract_colorFeatures(imread(filenameS))];
      colorFeaturesT = [colorFeaturesT; extract_colorFeatures(imread(filenameT))];
     MorphoFeaturesN = [MorphoFeaturesN; extract_colorFeatures(imread(filenameN))];
     MorphoFeaturesS = [MorphoFeaturesS; extract_colorFeatures(imread(filenameS))];
     MorphoFeaturesT = [MorphoFeaturesT; extract_colorFeatures(imread(filenameT))];
end

colorFeature = [colorFeaturesN; colorFeaturesS; colorFeaturesT];
TextureFeature = [featureN; featureS; featureT];
MorphoFeatures = [MorphoFeaturesN; MorphoFeaturesS; MorphoFeaturesT];
scolorFeature = size(colorFeature);
sTextureFeature = size(TextureFeature);
sMorphoFeatures = size(MorphoFeatures);
a = max(max(scolorFeature(1),sTextureFeature(1)),sMorphoFeatures(1));
total = [[colorFeature;zeros(abs([a, 0]-scolorFeature))],[TextureFeature;zeros(abs([a,0]-sTextureFeature))], [MorphoFeatures;zeros(abs([a,0]-sMorphoFeatures))]];

save("featureTexturer.mat", "total");
