
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

# import some data to play with

import os
import pickle
import config
import dnnlib
import gzip
import json
import numpy as np
#from tqdm import tqdm_notebook
import warnings
import matplotlib.pylab as plt
import dnnlib.tflib as tflib
import PIL.Image

warnings.filterwarnings("ignore")
tflib.init_tf()

keys = [x for x in range(-30,40,10)]
dic_distances = {}
for key in keys:
    dic_distances[key] = [i for i in range(key-4,key+5)]

def return_label(json_label,dic_distances):
    yaw = json_label['faceAttributes']['headPose']['yaw']
    yaw = round(yaw)
    for key,yaws_in_key in dic_distances.items():
        if yaw in yaws_in_key:
            return key
    return -1



LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'

with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

labels_data[0]  # wow. many fields. amaze

X_data = dlatent_data.reshape((-1, 18*512))
y_yaw = np.array([return_label(x,dic_distances) for x in labels_data])

not_wanted_indices = np.where(y_yaw==-1)[0]

filtered_X_data = np.delete(X_data, not_wanted_indices,axis=0)
filtered_Y_yaw = np.delete(y_yaw, not_wanted_indices)
filtered_labels = [i for j, i in enumerate(labels_data) if j not in not_wanted_indices]




real_y_yaws = np.array([x['faceAttributes']['headPose']['yaw'] for x in labels_data])
linreg = LinearRegression()
linreg.fit(X_data,real_y_yaws)
directions = linreg.coef_
print(directions)
print(directions.shape)
linreg.predict((X_data[:10]))
print(real_y_yaws[:10])
np.save('wLinReg.npy', directions)

exit()

# prediction = logreg.predict(directions.reshape((7,18*512)))
# print(prediction)

indexes_of_yaw0 = np.where(filtered_Y_yaw==0)[0]
indexes_of_yaw0 = indexes_of_yaw0[:20]

url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

a = np.zeros(shape=(1, 18, 512))

for index in indexes_of_yaw0:
    latent_yaw0 = filtered_X_data[index]
    a[0] = latent_yaw0.reshape((18, 512))
    images = Gs.components.synthesis.run(a, minibatch_size=1, randomize_noise=False,
                                         output_transform=dict(func=tflib.convert_images_to_uint8,
                                                               nchw_to_nhwc=True), structure='fixed')
    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, f'img_index_{index}_original.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    for coeff in range(-20,21):
        if coeff < 0:
            direction = directions2[1]
            direction_type = "+20"
        else:
            direction = directions2[5]
            direction_type = "-20"
        latent_yaw0 = filtered_X_data[index] + direction
        a[0] = latent_yaw0.reshape((18,512))
        images = Gs.components.synthesis.run(a, minibatch_size=1, randomize_noise=False,
                                            output_transform=dict(func=tflib.convert_images_to_uint8,
                                                                  nchw_to_nhwc=True), structure='fixed')
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, f'img_index_{index}_direction_{direction_type}_coeff_{0.1*coeff}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


print("hi")
