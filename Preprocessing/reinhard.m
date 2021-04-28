function normalized_img = reinhard(img, target_img)
    lab = rgb2lab(img);
    new_lab = zeros(size(lab));
    for i = 1:3
        lab_mean = mean2(lab(:,:,i));
        lab_std = std2(lab(:,:,i));
        target_mean = mean2(target_img(:,:,i));
        target_std = std2(target_img(:,:,i));
        new_lab(:,:,i) = ((lab(:,:,i)-lab_mean)/lab_std*target_std) + target_mean;
    end
    normalized_img = lab2rgb(new_lab);
end