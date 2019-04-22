import os
import pandas as pd
import numpy as np
import cv2
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Activation, BatchNormalization, MaxPooling2D, Conv2D, Conv2DTranspose, concatenate
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from datetime import datetime
from matplotlib import pyplot as plt


def train_val_split_dir(directory):
    df_train = pd.read_csv(os.path.join(directory, 'train_masks.csv'))
    ids_train = [n.split('.')[0] for n in df_train['img']]

    x_train = []
    y_train = []
    for i in ids_train:
        x_train.append(os.path.join(directory, 'train/{}.jpg'.format(i)))
        y_train.append(os.path.join(directory, 'train_masks/{}_mask.gif'.format(i)))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    print("Number of training examples: {}".format(len(x_train)))
    print("Number of validation examples: {}".format(len(x_val)))

    return x_train, x_val, y_train, y_val


# def load_image_and_mask(image_dir, mask_dir, normalize=True):
#     img_str = tf.read_file(image_dir)
#     image = tf.image.decode_jpeg(img_str, channels=3)
#
#     label_img_str = tf.read_file(mask_dir)
#     label_img = tf.image.decode_gif(label_img_str)
#     # These are gif images so they return as (num_frames, h, w, c)
#     # The label image should only have values of 1 or 0, indicating pixel wise object (car) or not (background).
#     # We take the first channel only.
#     label_img = label_img[0, :, :, 0]
#     mask = tf.expand_dims(label_img, axis=-1)
#
#     if normalize:
#         image = image/255
#         mask = mask/255
#
#     return image, mask
#
#
# def flip_img(image, mask, horizontal_flip=False):
#     if horizontal_flip:
#         flip_prob = tf.random_uniform([], 0, 1)
#         image, mask = tf.cond(tf.less(flip_prob, 0.5),
#                               lambda: (tf.image.flip_left_right(image), tf.image.flip_left_right(mask)),
#                               lambda: (image, mask))
#
#     return image, mask
#
#
# def augmentation(image, mask,
#                  resize=None,  # Resize the image to some size e.g. [256, 256]
#                  hue_delta=None,  # Adjust the hue of an RGB image by random factor
#                  horizontal_flip=False,  # Random left right flip,
#                  ):
#     if resize is not None:
#         # Resize both images
#         image = tf.image.resize_images(image, resize)
#         mask = tf.image.resize_images(mask, resize)
#
#     if hue_delta is not None:
#         image = tf.image.random_hue(image, hue_delta)
#
#     image, mask = flip_img(image, mask, horizontal_flip=horizontal_flip)
#
#     return image, mask


def construct_dataset(image_dir, mask_dir, target_size=(64, 64), identification='train', save=False):
    resize = target_size
    # read image to array
    image = cv2.imread(image_dir[0], -1)
    # resize the image
    image = cv2.resize(image, resize)
    # expand the dimension
    image = image[np.newaxis, :, :, :]

    mask = plt.imread(mask_dir[0])
    mask = cv2.resize(mask, resize)
    mask = mask[np.newaxis, :, :, 0]
    mask = mask[:, :, :, np.newaxis]

    for i, (img_dir, msk_dir) in enumerate(zip(image_dir[1:], mask_dir[1:])):
        # read image to array
        image_2 = cv2.imread(img_dir, -1)
        # resize the image
        image_2 = cv2.resize(image_2, resize)
        # expand the dimension
        image_2 = image_2[np.newaxis, :, :, :]

        mask_2 = plt.imread(msk_dir)
        mask_2 = cv2.resize(mask_2, resize)
        mask_2 = mask_2[np.newaxis, :, :, 0]
        mask_2 = mask_2[:, :, :, np.newaxis]

        image = np.append(image, image_2, axis=0)
        mask = np.append(mask, mask_2, axis=0)

        print('processing: ' + str(i))

    print("train/val shape: " + str(image.shape))
    print("train/val shape: " + str(mask.shape))

    if save:
        if not os.path.exists('carvana_dataset'):
            os.mkdir('carvana_dataset')
        h5f = h5py.File('carvana_dataset/{}.h5'.format(identification), 'w')
        h5f.create_dataset('image'.format(identification), data=image)
        h5f.create_dataset('mask'.format(identification), data=mask)

    return image, mask


def load_dataset():
    train_dataset = h5py.File('carvana_dataset/train.h5', 'r')
    image_train = np.array(train_dataset['image'][:])  # train set features
    mask_train = np.array(train_dataset['mask'][:])  # train set labels

    val_dataset = h5py.File('carvana_dataset/val.h5', 'r')
    image_val = np.array(val_dataset['image'][:])  # test set features
    mask_val = np.array(val_dataset['mask'][:])  # test set labels

    return image_train, mask_train, image_val, mask_val


def my_model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    x_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    encoder0 = Activation('relu')(x)
    # MAXPOOL
    x = MaxPooling2D((2, 2), strides=(2, 2))(encoder0)

    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    encoder1 = Activation('relu')(x)
    # MAXPOOL
    x = MaxPooling2D((2, 2), strides=(2, 2))(encoder1)

    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    encoder2 = Activation('relu')(x)
    # MAXPOOL
    x = MaxPooling2D((2, 2), strides=(2, 2))(encoder2)

    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    encoder3 = Activation('relu')(x)
    # MAXPOOL
    x = MaxPooling2D((2, 2), strides=(2, 2))(encoder3)

    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    encoder4 = Activation('relu')(x)
    # MAXPOOL
    x = MaxPooling2D((2, 2), strides=(2, 2))(encoder4)

    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    center = Activation('relu')(x)

    # decoder
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(center)
    x = concatenate([encoder4, x], axis=-1)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([encoder3, x], axis=-1)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([encoder2, x], axis=-1)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([encoder1, x], axis=-1)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([encoder0, x], axis=-1)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    # 1x1 conv, sigmoid to output 0/1
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x_output = Activation(activation='sigmoid')(x)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=x_input, outputs=x_output, name='MyModel')
    model.summary()

    return model


def config_cpu():
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count={'CPU': 4})
    session = tf.Session(config=config)
    K.set_session(session)


def train_test_model(x_train, y_train, x_test, y_test, epochs, batch_size, norm=True, save=False):
    """
    You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
    1. Create the model by calling the function above
    2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
    3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
    4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`
    """

    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if norm:
        # normalization performs on mask as well, because the labels are 0 and 255 (not 0 and 1)
        x_train, y_train, x_test, y_test = normalization(x_train, y_train, x_test, y_test)

    tensorboard = TensorBoard(log_dir='carvana_dataset/logs/size{}batch{}epoch{}_'.format(str(x_train.shape[1:-1]),
                                                                                          str(batch_size),
                                                                                          str(epochs)) +
                                      datetime.now().strftime('%m-%d-%H-%M'))

    model = my_model(x_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size,
                        callbacks=[tensorboard])

    pre = model.evaluate(x=x_test, y=y_test)
    print('Loss = ' + str(pre[0]))
    print('Test Accuracy = ' + str(pre[1]))

    if save:
        save_model(model)

    return history


def normalization(image_train, mask_train, image_val, mask_val):
    # Normalize image vectors
    image_train = image_train / 255.
    mask_train = mask_train / 255.
    image_val = image_val / 255.
    mask_val = mask_val / 255.

    return image_train, mask_train, image_val, mask_val


def save_model(model):
    model.save('carvana_dataset/MyModel.h5')
    plot_model(model, to_file='carvana_dataset/MyModel.pdf')


def load_my_model():
    model = load_model('carvana_dataset/MyModel.h5')

    return model


def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])

    return image


def prediction_visual(model, x_test, y_test, which_show=0, save=False):
    _, _, x_test, y_test = normalization(0, 0, x_test, y_test)
    prediction = model.predict(x_test)

    num = np.random.choice(np.arange(x_test.shape[0]), 1)[0]

    fig, axs = plt.subplots(2, 4, figsize=(10, 6), sharey='row')

    axs[0, 0].imshow(bgr_to_rgb(x_test[which_show, :, :, :]), label='input image')
    axs[0, 0].set_title('input car image')
    axs[0, 1].imshow(y_test[which_show, :, :, 0], cmap='gray', label='ground_truth')
    axs[0, 1].set_title('true mask')

    axs[1, 0].imshow(bgr_to_rgb(x_test[which_show, :, :, :]), label='input image')
    axs[1, 0].set_title('input car image')
    axs[1, 1].imshow(prediction[which_show, :, :, 0], cmap='gray', label='prediction')
    axs[1, 1].set_title('predicted mask')

    axs[0, 2].imshow(bgr_to_rgb(x_test[num, :, :, :]), label='input image')
    axs[0, 2].set_title('input car image')
    axs[0, 3].imshow(y_test[num, :, :, 0], cmap='gray', label='ground_truth')
    axs[0, 3].set_title('true mask')

    axs[1, 2].imshow(bgr_to_rgb(x_test[num, :, :, :]), label='input image')
    axs[1, 2].set_title('input car image')
    axs[1, 3].imshow(prediction[num, :, :, 0], cmap='gray', label='prediction')
    axs[1, 3].set_title('predicted mask')

    if save:
        fig.savefig('carvana_dataset/prediction_visualization.pdf')


def learning_visual(history, save=False):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex='col')
    # fig.subplots_adjust(hspace=0.2, wspace=0.4, left=0.07, right=0.96, bottom=0.1, top=0.95)

    # Plot training & validation accuracy values
    axs[0].plot(history.history['acc'], label='accuracy', color='C0')
    # plt.plot(history.history['val_acc'])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'], label='loss', color='C1')
    # plt.plot(history.history['val_loss'])
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper left')

    if save:
        fig.savefig('carvana_dataset/learning_curve.pdf')

