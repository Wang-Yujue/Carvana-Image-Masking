from carvana_image_masking import *


config_cpu()

directory = '/Users/wangyujue/Documents/GitHub/models/samples/outreach/blogs/segmentation_blogpost/' \
            'carvana-image-masking-challenge'

# resize = (128, 128)
# x_train, x_val, y_train, y_val = train_val_split_dir(directory)
# construct_dataset(x_train, y_train, target_size=resize, identification='train', save=True)
# construct_dataset(x_val, y_val, target_size=resize, identification='val', save=True)

image_train, mask_train, image_val, mask_val = load_dataset()
# batch_size = 64
# epochs = 10
#
# history = train_test_model(image_train, mask_train, image_val, mask_val, epochs, batch_size, norm=True, save=True)
# learning_visual(history, save=True)

model = load_my_model()
prediction_visual(model, image_val, mask_val, which_show=0, save=True)

plt.show()
