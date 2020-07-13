from skimage import feature
from skimage.transform import resize

def extract_feature_hog(image, orientations = 9, pixels_per_cell = (4, 4), cells_per_block = (2, 2), transform_sqrt = True, feature_vector = True, visualize = False):
  image = resize(image, (28, 28))
  return feature.hog(image, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, visualize = visualize, transform_sqrt = transform_sqrt, feature_vector = feature_vector)