Aa(:, :, 1) = [1, 2; 3, 4; 5, 6];
Aa(:, :, 2) = [11, 12; 13, 14; 15, 16];
Aa(:, :, 3) = [21, 22; 23, 24; 25, 26];
Aa(:, :, 4) = [31, 32; 33, 34; 35, 36];
Aa_p = permute(Aa, [2, 1, 3]);
newAa = reshape(permute(Aa_p(:), [2, 1, 3]), [6 4]);

Aab = [1, 2; 3, 4; 5, 6];
Aabdf = permute(Aab, [2, 1]);
newAab = permute(Aabdf(:), [2, 1]);
