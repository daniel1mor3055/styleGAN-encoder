from sklearn.linear_model import LogisticRegression
import os
import pickle
import config
import dnnlib
import gzip
import numpy as np
import warnings
import dnnlib.tflib as tflib
import PIL.Image

warnings.filterwarnings("ignore")
tflib.init_tf()


LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

# reshape d-latents from (18,512) to (1,18*512)
X_data = dlatent_data.reshape((-1, 18 * 512))

# label data accordingly to label function
y_gender = np.array([x['faceAttributes']['gender'] == 'male' for x in labels_data])

"""Train and save directions"""
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', verbose=1, max_iter=40000)
logreg.fit(X_data, y_gender)
male_direction = logreg.coef_
""" Saved learned directions"""
np.save('male_direction.npy', male_direction)

""" Load learned directions """
# male_direction = np.load('male_direction.npy')

"""Load models"""
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

""" Let's start playing with some data. """


def apply_masculine_and_generate(Gs, male_direction, x_data):
    for coeff in range(-20, 21):
        wanted_direction = 0.1 * coeff * male_direction
        x_data_moved_in_male_direction = x_data + wanted_direction
        x_data_moved_in_male_direction = x_data_moved_in_male_direction.reshape((-1, 18, 512))
        images = Gs.components.synthesis.run(x_data_moved_in_male_direction, minibatch_size=1, randomize_noise=False,
                                             output_transform=dict(func=tflib.convert_images_to_uint8,
                                                                   nchw_to_nhwc=True), structure='fixed')
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir,
                                    f'img_index_{index}_direction_male_coeff_{0.1*coeff}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


# take indices of 20 pictures of specified yaw value
indexes_of_females = np.where(y_gender == 0)[0][:20]
for index in indexes_of_females:
    # Generate original image & save
    x_data_female = X_data[index].reshape((-1, 18, 512))
    images = Gs.components.synthesis.run(x_data_female, minibatch_size=1, randomize_noise=False,
                                         output_transform=dict(func=tflib.convert_images_to_uint8,
                                                               nchw_to_nhwc=True), structure='fixed')
    # Save original image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, f'img_index_{index}_original.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    # Create different poses & save them.
    apply_masculine_and_generate(Gs, male_direction, X_data[index])
