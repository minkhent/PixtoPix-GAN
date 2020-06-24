from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint

def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'],data['arr_1']
    X1 = (X1 - 127.5)/ 127.5
    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]

def plot_images(src_image, gen_image, tar_image):
    images = vstack((src_image, gen_image, tar_image))
    images = (images + 1) / 2.0

    titles= ['Source', 'Generated', 'Expected']

    for i in range(len(images)):
        pyplot.subplot(1, 3, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(images[i])
        pyplot.title(titles[i])
    pyplot.show()

#load compressed data
[X1, X2] = load_real_samples('data/maps_256.npz')
print('Loaded', X1.shape, X2.shape)

# load model
model = load_model('model/model_05000.h5')

index = randint(0, len(X1),1)

print(index)

src_image, tar_image = X1[index], X2[index]

gen_image = model.predict(src_image)

plot_images(src_image, gen_image, tar_image)







