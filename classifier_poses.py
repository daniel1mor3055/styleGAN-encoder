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

yaws_classes = [x for x in range(-30, 40, 10)]
# [yaw-4,yaw+4] will be mapped to yaw
yaws_label_mapping = {}
for yaw in yaws_classes:
    yaws_label_mapping[yaw] = [i for i in range(yaw - 4, yaw + 5)]


def return_pose_label(json_label):
    # all points that with equal distance to 2 different classes,
    # is labeled with -1. (meaning it will be removed from dataset)
    yaw = json_label['faceAttributes']['headPose']['yaw']
    yaw = round(yaw)
    for yaw_class, yaws_in_class in yaws_label_mapping.items():
        if yaw in yaws_in_class:
            return yaw_class
    return -1


LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

# reshape d-latents from (18,512) to (1,18*512)
X_data = dlatent_data.reshape((-1, 18 * 512))

# label data accordingly to label function
y_yaw = np.array([return_pose_label(x) for x in labels_data])

# remove from datasets not wanted data samples.
# every datapoint that received -1 for his label, is removed.
not_wanted_indices = np.where(y_yaw == -1)[0]

filtered_X_data = np.delete(X_data, not_wanted_indices, axis=0)
filtered_Y_yaw = np.delete(y_yaw, not_wanted_indices)
filtered_labels = [label for index, label in enumerate(labels_data) if index not in not_wanted_indices]

"""Let's plot the statistics of our data and labels"""
# from collections import Counter
# w = Counter(filtered_Y_yaw)
# plt.bar(w.keys(), w.values())
# plt.show()

"""More efforts on trying to understand data distributions"""
# y_yaw_data = np.array([return_pose_label(x) for x in filtered_labels])
# y_gender_data = np.array([x['faceAttributes']['gender'] == 'male' for x in filtered_labels]) # sry girls :(
#
# plt.hist(y_yaw_data[y_gender_data], bins=30, color='red', alpha=0.5, label='male')
# plt.hist(y_yaw_data[~y_gender_data], bins=30, color='blue', alpha=0.5, label='female')
# plt.legend()
# plt.title('Distribution of yaw within gender')
# plt.xlabel('yaw')
# plt.ylabel('Population')
# plt.show()
# exit()

"""Train and save directions"""
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial',verbose=1,max_iter=40000)
logreg.fit(filtered_X_data,filtered_Y_yaw)
pose_directions = logreg.coef_
""" Saved learned directions"""
np.save('pose_directions.npy', pose_directions)

""" Load learned directions """
# pose_directions = np.load('pose_directions.npy')

"""Load models"""
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

""" Let's start playing with some data. """

def generate_different_poses(Gs, pose_directions,x_data):
    for coeff in range(-20, 21):
        if coeff < 0:
            wanted_direction = -0.1*coeff*pose_directions[1]
            direction_type = "-20"
        else:
            wanted_direction = 0.1*coeff*pose_directions[5]
            direction_type = "+20"
        x_data_with_yaw_moved = x_data + wanted_direction
        x_data_with_yaw_moved = x_data_with_yaw_moved.reshape((-1,18, 512))
        images = Gs.components.synthesis.run(x_data_with_yaw_moved, minibatch_size=1, randomize_noise=False,
                                             output_transform=dict(func=tflib.convert_images_to_uint8,
                                                                   nchw_to_nhwc=True), structure='fixed')
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir,
                                    f'img_index_{index}_direction_{direction_type}_coeff_{0.1*coeff}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

# take indices of 20 pictures of specified yaw value
indexes_of_wanted_yaw = np.where(filtered_Y_yaw == 0)[0][:20]
for index in indexes_of_wanted_yaw:
    # Generate original image & save
    x_data_with_wanted_yaw = filtered_X_data[index].reshape((-1,18, 512))
    images = Gs.components.synthesis.run(x_data_with_wanted_yaw, minibatch_size=1, randomize_noise=False,
                                         output_transform=dict(func=tflib.convert_images_to_uint8,
                                                               nchw_to_nhwc=True), structure='fixed')
    # Save original image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, f'img_index_{index}_original.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    # Create different poses & save them.
    generate_different_poses(Gs,pose_directions,filtered_X_data[index])

