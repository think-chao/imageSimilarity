import os
import random
from sklearn.decomposition import PCA
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

model = keras.applications.VGG16(weights='imagenet', include_top=True)
model.summary()


def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def get_closest_images(query_image_idx, num_results=5):
    distances = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results + 1]
    return idx_closest


def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


# img, x = load_image("C:/Users/king/Pictures/成都/12.jpg")
# print("shape of x: ", x.shape)
# print("data type: ", x.dtype)
# plt.imshow(img)
# plt.show()
#
# # predictions = model.predict(x)
# # for _, pred, prob in decode_predictions(predictions)[0]:
# #     print("predicted %s with probability %0.3f" % (pred, prob))
#
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
# feat_extractor.summary()
#
# feat = feat_extractor.predict(x)
#
# plt.figure(figsize=(16, 4))
# plt.plot(feat[0])
# plt.show()

images_path = 'E:/file/data/101_ObjectCategories/101_ObjectCategories/watch'
image_extensions = ['.jpg', '.png', '.jpeg']  # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if
          os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

print("keeping %d images to analyze" % len(images))

features = []
for i, image_path in enumerate(images):
    if i % 1000 == 0:
        print("analyzing image %d / %d" % (i, len(images)))
    img, x = load_image(image_path)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

print('finished extracting features for %d images' % len(images))

features = np.array(features)
pca = PCA()
pca.fit(features)

pca_features = pca.transform(features)

# do a query on a random image
query_image_idx = int(len(images) * random.random())
idx_closest = get_closest_images(query_image_idx)
print(idx_closest)
query_image = get_concatenated_images([query_image_idx], 300)
results_image = get_concatenated_images(idx_closest, 200)
# display the query image
plt.figure(figsize = (5,5))
plt.imshow(query_image)
plt.title("query image (%d)" % query_image_idx)

# display the resulting images
plt.figure(figsize = (16,12))
plt.imshow(results_image)
plt.title("result images")
plt.show()