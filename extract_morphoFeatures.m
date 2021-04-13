function morphoFeatures = extract_morphoFeatures(img)
se = offsetstrel('ball',3, 3);
erodedImg = imerode(img, se);
[centers,radii] = imfindcircles(erodedImg,[5 15],'ObjectPolarity','dark', ...
    'Sensitivity',0.85);
level = graythresh(erodedImg);
BW = imbinarize(erodedImg, level);
s = regionprops(BW, 'Centroid');
morphoFeatures = [length(centers), length(s)];
end