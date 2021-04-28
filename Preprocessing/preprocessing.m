% We have 300 inpute images in total; 100 Nerosis, 100 Storma, and 100
% Tumor. Here, it considers the first iamges of each category as target
% images.
path_directory = 'Image_data'; 
original_files = dir([path_directory '/*.png']);
%%

rand_crop = randi([1 100], 1, 20); % Array containing random integers
rand_flip = randi([1 100], 1, 20);
rand_rot = randi([1 100], 1, 20);
target_necrosis = rgb2lab(imread("Image_data/Necrosis_1.png")); % Target image for Necrosis
target_stroma = rgb2lab(imread("Image_data/Stroma_1.png")); % Target image for Stroma
target_tumor = rgb2lab(imread("Image_data/Tumor_1.png")); % Target image for Tumor
labels = []; % Labels for the future use (Module 3)
labels_crop = [];
%%
% Read, normalize, randomly crop, flip, and rotate the images
for k=2:100 % Skip the first image since it's target
    filenameN = [path_directory '/' original_files(k).name]; % Get image name for Necrosis Image
    filenameS = [path_directory '/' original_files(k+100).name]; % Get image name for Stroma Image
    filenameT = [path_directory '/' original_files(k+200).name]; % Get image name for Tumor Image
    input_normN = reinhard(imread(filenameN), target_necrosis); % Normalize Necrosis Image
    input_normS = reinhard(imread(filenameS), target_stroma); % Normalize Stroma Image
    input_normT = reinhard(imread(filenameT), target_tumor); % Normalize Stroma Image
    imwrite(input_normN, "Image_data_normalized/Necrosis/Necrosis_" + k + ".png");
    imwrite(input_normS, "Image_data_normalized/Stroma/Stroma_" + k + ".png");
    imwrite(input_normT, "Image_data_normalized/Tumor/Tumor_" + k + ".png");
    if ismember(k, rand_crop) % Check if k is in rand_crop array
        rect = randomWindow2d(size(input_normN), [244 244]); % Randomly generate ROI
        imwrite(imcrop(input_normN, rect), "Image_data_normalized/Necrosis/Necrosis_crop_" + k + ".png");
        imwrite(imcrop(input_normS, rect), "Image_data_normalized/Stroma/Stroma_crop_" + k + ".png");
        imwrite(imcrop(input_normT, rect), "Image_data_normalized/Tumor/Tumor_crop_" + k + ".png");
    if ismember(k, rand_flip) % Check if k is in rand_flip
        flip_num = randi([1 2], 1); % Randomly choose Horizontal/Vertical
        imwrite(flip(input_normN, flip_num), "Image_data_normalized/Necrosis/Necrosis_flip_" + k + ".png");
        imwrite(flip(input_normS, flip_num), "Image_data_normalized/Stroma/Stroma_flip_" + k + ".png");
        imwrite(flip(input_normT, flip_num), "Image_data_normalized/Tumor/Tumor_flip_" + k + ".png");
    if ismember(k, rand_rot)
        rot_num = randi([1 3], 1); % Randomly choose 90/180/270
        imwrite(imrotate(input_normN, 90*rot_num), "Image_data_normalized/Necrosis/Necrosis_rot_" + k + ".png");
        imwrite(imrotate(input_normS, 90*rot_num), "Image_data_normalized/Stroma/Stroma_rot_" + k + ".png");
        imwrite(imrotate(input_normT, 90*rot_num), "Image_data_normalized/Tumor/Tumor_rot_" + k + ".png");
    end
    end
    end
end
%%