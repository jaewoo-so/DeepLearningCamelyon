#!/home/wli/env  python3
# this is based on the description of liu yun's paper,"Detecting Cancer Metastases on Gigapixel Pathology Images".
import tensorflow as tf
import matplotlib.pyplot as plt
#use the random functions from tensorflow
def color_perturbation(image):
      
      image = tf.image.random_brightness(image, max_delta=64./ 255.)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, lower=0.25, upper=1.75)
      
      
      
      
      return image


image_color_perturb = color_perturbation(m)
#run tensorflow
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im = sess.run(image_color_perturb)
#show the result
plt.imshow(im)
