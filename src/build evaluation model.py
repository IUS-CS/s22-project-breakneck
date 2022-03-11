from keras.preprocessing.ImageLoader import load_img
from keras.preprocessing.ImageLoader import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16();
# load an ImageLoader from file
ImageLoader = ImageLoader.resize((256, 256), ImageLoader.BILINEAR) ;
mlt.imshow(ImageLoader);
ImageLoader = load_img('ImageLoader', target_size=(256, 256));
ImageLoader = img_to_array(ImageLoader);
ImageLoader = preprocess_input(ImageLoader);
label = label[1000][1000];
print('%s (%.2f%%)' % (label[1], label[2]*1000));