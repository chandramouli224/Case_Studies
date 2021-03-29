import keras
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
from tensorflow.keras import layers
import cv2
def import_model():
    mobile = keras.applications.mobilenet.MobileNet()
    return mobile
def prepare_image(frame):
    img_path = ''
    #frame = image.load_img(frame , target_size=(224, 224))
    #print(frame.size)
    frame=frame.reshape(1,224, 224,3)
    # img_array = image.img_to_array(frame)
    # img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    return frame
# preprocessed_image = prepare_image('D:\Lectures\casestudy\dataset\German_Shepherd.jpeg')
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)

def build_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNet(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(224, 224, 3),
        pooling='avg')
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers

    op = Dense(224, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(1, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

def train_model():
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    rescaling =  layers.experimental.preprocessing.Rescaling(1./255)
    # base_model = MobileNet(
    #     include_top=False,
    #     weights='imagenet',
    #     input_shape=(224, 224, 3)
    # )  # Do not include the ImageNet classifier at the top (1000 => 2)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # base_model.trainable = False
    # inputs = tf.keras.Input(shape=(224, 224, 3))
    x=base_model.output
    # inputs=base_model.input
    # inputs = base_model.input
    # # Image augmentation block
    x = data_augmentation(x)
    x= preprocess_input(x)
    # # Entry block
    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x= base_model(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(224,activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(42), bias_initializer='RandomNormal')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(224,activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(42), bias_initializer='RandomNormal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(224, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(42),
                              bias_initializer='RandomNormal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(224, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(42),
                              bias_initializer='RandomNormal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_uniform', bias_initializer='RandomNormal')(x)
    model = tf.keras.models.Model(inputs=base_model.inputs,outputs = predictions)
    optimizer = tf.keras.optimizers.Adam(lr=0.00001)
    # loss = "categorical_crossentropy"
    fine_tune_at = 100
    base_model.trainable = True
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizer,
        # optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.summary()
    return model