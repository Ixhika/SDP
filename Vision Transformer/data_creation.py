import Augmentor

p = Augmentor.Pipeline("DATASET/test/tooth_brush")
# doublemint mouse suns_cream


p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.3)
p.zoom(probability=0.5, min_factor=0.6, max_factor=1.0)
p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
p.skew(probability=0.4, magnitude=0.5)
p.random_distortion(probability=0.3, grid_width=6, grid_height=6, magnitude=4)
p.gaussian_distortion(probability=0.2, grid_width=5, grid_height=5, magnitude=2, corner='bell', method='in')
# p.random_erasing(probability=0.2, rectangle_area=0.05)  # Optional: disable for no patches
p.crop_random(probability=0.4, percentage_area=0.8)
p.resize(probability=1.0, width=224, height=224)

p.sample(192)

