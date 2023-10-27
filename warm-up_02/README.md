# Pixel matching between a pair of stereo image
## Steps
1. Calculate the fundamental matrix F linking the two images
2. Once F is calculated, compute the corresponding epipolar line
3. Find the best ZNCC score with a limited range along the line.
