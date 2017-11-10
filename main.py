# Python

'''
Andrea Pasini
References to cite github:
https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow
https://github.com/hszhao/PSPNet
'''


from scipy import misc, ndimage
from pspnet import PSPNet50
from pspnet import predict_multi_scale
import numpy as np
import utils
from os.path import splitext, join, isfile





EVALUATION_SCALES = [1.0]  # must be all floats!
#if args.multi_scale:
#    EVALUATION_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # must be all floats!
#    EVALUATION_SCALES = [0.15, 0.25, 0.5]  # must be all floats!


sliding = False
flip = False









from ade20k_labels import labels

id = {label.id for label in labels if label.name=='ceiling'}
#todo get label and idddddd












img = misc.imread('PSPNet-Keras-tensorflow-master/example_images/ade20k2.jpg')

#create the neural network
pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights='pspnet50_ade20k')

#Run over input data
class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, sliding, flip)








print("Writing results...")

class_image = np.argmax(class_scores, axis=2)   #Get best classes (highest score!!)
pm = np.max(class_scores, axis=2)
colored_class_image = utils.color_class_image(class_image, 'ade20k')
# colored_class_image is [0.0-1.0] img is [0-255]
alpha_blended = 0.5 * colored_class_image + 0.5 * img

#filename, ext = splitext('PSPNet-Keras-tensorflow-master/example_results/')

#misc.imsave(filename + "_seg" + ext, colored_class_image)
#misc.imsave(filename + "_probs" + ext, pm)
#misc.imsave(filename + "_seg_blended" + ext, alpha_blended)

misc.imshow(colored_class_image)