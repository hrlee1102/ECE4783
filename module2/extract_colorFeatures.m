function color_features = extract_colorFeatures(img)
[yRed, x] = imhist(img(:,:,1), 16);
[yGreen, x] = imhist(img(:,:,2), 16);
[yBlue, x] = imhist(img(:,:,3), 16);
rAvg = mean2(img(:,:,1));
gAvg = mean2(img(:,:,2));
bAvg = mean2(img(:,:,3));
hsv = rgb2hsv(img);
iAvg = mean2(hsv(:,:,3));
color_features = [yRed.', yGreen.', yBlue.', rAvg, gAvg, bAvg, iAvg];
end