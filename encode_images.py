import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import tensorflow as tf
import requests
import json

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl


def split_to_batches(l, n):
#splits input to batches - will be used for spliting ref_images into batches of wanted batch size
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=100, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))
	#ref_images is a list of all files in src dir 
	
    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise) #creates a generator using Gs_network.
	
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

##    subscription_key = "e34b0865d37342ba8b3fc5a846aac275"
##
##    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
##
##    # image_url = 'https://upload.wikimedia.org/wikipedia/commons/3/37/Dagestani_man_and_woman.jpg'
##    # image_url = 'https://www.contrareplica.mx/uploads/2019/07/19/normal/00c043cf234470c3739b2521ff0e18b7.jpg'
##    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': subscription_key}
##
##    params = {
##        'returnFaceId': 'true',
##        'returnFaceLandmarks': 'false',
##        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
##    }

# Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
	#the for loop goes over ref_images in batches of size that's equal to batch_size. 
        perceptual_model.set_reference_images(images_batch) #initializes features_weight and ref_img_features(corresponding to current images_batch) of perceptual model
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, learning_rate=args.lr) #pass generator.dlatent_variable for optmizer to update 
        pbar = tqdm(op, leave=False, total=args.iterations) #progress bar
        i = 0
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
            if(i%250==0):
                print("\nSaving image!!!!\n")
                generated_images = generator.generate_images()
                generated_dlatents = generator.get_dlatents()
                for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
                    img = PIL.Image.fromarray(img_array, 'RGB')
                    img.save(os.path.join(args.generated_images_dir, f'{img_name}_iteration_{i}.png'), 'PNG')
##                    np.save(os.path.join(args.dlatent_dir, f'{img_name}_iteration_{i}.npy'), dlatent)
            i = i + 1
        print(' '.join(names), ' loss:', loss)

        # Generate images from found dlatents and save them
        generated_images = generator.generate_images() #gets generator.generated_images after last optimizer iteration
        generated_dlatents = generator.get_dlatents() #gets generator.dlatent_variable after last optimizer iteration
        
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}_last_iteration.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}_last_iteration.npy'), dlatent)

##            path = f'/specific/netapp5_2/gamir/advml19/danielmor1/machinelearning/stylegan-encoder/generated_images/{img_name}.png'
##            data = open(path, 'rb')
##            response = requests.post(face_api_url, params=params,
##                                     headers=headers, data=data)
##
##            with open(
##                    f'/specific/netapp5_2/gamir/advml19/danielmor1/machinelearning/stylegan-encoder/labels/{img_name}.txt',
##                    'w') as outfile:
##                json.dump(response.json(), outfile)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
