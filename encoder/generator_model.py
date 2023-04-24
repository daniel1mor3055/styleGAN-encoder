import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
#returns a tf session constant tensor with 0 in all of its enteries. It'll get its shape from batch_size - a rank one tensor with batch_size entries(or a zero rank tensor if batch_size is 1)
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size):
#returns the variable in the tf session with the parameters specified below if it exists, otherwises creates it  and initializes it using a normally sampled random tensor
    return tf.get_variable('learnable_dlatents',
                           shape=(batch_size, 18, 512),
                           dtype='float32',
                           initializer=tf.initializers.random_normal())


class Generator:
    def __init__(self, model, batch_size, randomize_noise=False):
	#recieves nvidia's GS_network as an argument. batch size and randomize_noise are also passed in cmnd line 
        self.batch_size = batch_size #set batch size

        self.initial_dlatents = np.zeros((self.batch_size, 18, 512)) #set initial synthesis input to all zeroes
        model.components.synthesis.run(self.initial_dlatents,
                                       randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                                       custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size),
                                                      partial(create_stub, batch_size=batch_size)],
                                       structure='fixed')
		#using passed model's(GS_network) synthesis component and runs it for the first time with zeroes as input. Also, creates learnable_dlatents vraiable if doesn't exist and the stub.

        self.sess = tf.get_default_session() #set environment so it's possible to run our operations(defined in the below mentioned graph) - will be the synthesis 
        self.graph = tf.get_default_graph() #a tf graph is a composition of nodes that represent tf computational operations - c = tf.matmul(a, b), for example(input and output are both a non negative number of tensors). So the graph is basically a collection of operaions

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name) #gets the tf session variable called learnable_dlatents
        self.set_dlatents(self.initial_dlatents) #set dlatents to initial_dlatents while initializing a Generator
        self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0') #gets the output tensor after synthesis.run 
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False) #using a way provided by nVidia's to convert the network's output into an image 
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8) #casting safely to tf.unit8 

    def reset_dlatents(self):
	#setting delatens to initial_dlatents again
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
	#if the sahpe of dlatents fits our wanted latent space shape, the Generator object's dlatent_variable is set to dlatents
        assert (dlatents.shape == (self.batch_size, 18, 512))
        self.sess.run(tf.assign(self.dlatent_variable, dlatents))

    def get_dlatents(self):
	#fetch current learnable_dlatents
        return self.sess.run(self.dlatent_variable)

    def generate_images(self, dlatents=None):
	 
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)
