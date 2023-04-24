from sklearn.linear_model import LogisticRegression
import os
import pickle
import config
import dnnlib
import gzip
import numpy as np
import warnings
import matplotlib.pylab as plt
import dnnlib.tflib as tflib
import PIL.Image
warnings.filterwarnings("ignore")

tflib.init_tf()

def return_glasess_label(json_label):
    # label 0: No Glasses
    # label 1: ReadingGlasses / Sunglasses / SwimmingGoggles
    label = json_label['faceAttributes']['glasses']
    if label == 'NoGlasses':
        return 0
    else:
        return 1

LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

# reshape d-latents from (18,512) to (1,18*512)
X_data = dlatent_data.reshape((-1, 18*512))

# label data accordingly to label function
y_glasses = np.array([return_glasess_label(x) for x in labels_data])

# remove from datasets not wanted data samples.
# every datapoint that received -1 for his label, is removed.
# not_wanted_indices = np.where(y_glasses == -1)[0]
#
# filtered_X_data = np.delete(X_data, not_wanted_indices, axis=0)
# filtered_Y_yaw = np.delete(y_glasses, not_wanted_indices)
# filtered_labels = [label for index, label in enumerate(labels_data) if index not in not_wanted_indices]

"""Let's plot the statistics of our data and labels"""
# labels = [x['faceAttributes']['glasses'] for x in labels_data]
# from collections import Counter
# w = Counter(labels)
# print(w)
# plt.bar(w.keys(), w.values())
# plt.show()
# exit()


"""Train and save directions"""
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial',verbose=1,max_iter=40000)
logreg.fit(X_data,y_glasses)
glasses_direction = logreg.coef_
""" Saved learned directions"""
np.save('glasses_direction.npy', glasses_direction)

""" Load learned directions """
glasses_direction = np.load('glasses_direction.npy')

"""Load models"""
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

""" Let's start playing with some data. """
indexes_of_noGlasses = np.where(y_glasses==1)[0][:20]
indexes_of_noGlasses = indexes_of_noGlasses[:20]
# take indices of 20 pictures of specified glasses class

def generate_different_glasses(Gs, glasses_direction,x_data):
    for i in range(-8,9):
        coeff = i*0.5
        x_data_moved = x_data + coeff*glasses_direction
        x_data_moved = x_data_moved.reshape((-1, 18, 512))
        images = Gs.components.synthesis.run(x_data_moved, minibatch_size=1, randomize_noise=False,
                                            output_transform=dict(func=tflib.convert_images_to_uint8,
                                                                  nchw_to_nhwc=True), structure='fixed')
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, f'img_index_{index}_coeff_{coeff}_glasses.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

for index in indexes_of_noGlasses:
    x_data_with_glasses_class = X_data[index].reshape((-1, 18, 512))
    images = Gs.components.synthesis.run(x_data_with_glasses_class, minibatch_size=1, randomize_noise=False,
                                         output_transform=dict(func=tflib.convert_images_to_uint8,
                                                               nchw_to_nhwc=True), structure='fixed')
    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, f'img_index_{index}_original.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    # add glasses direction with different coeffs
    generate_different_glasses(Gs,glasses_direction,X_data[index])

