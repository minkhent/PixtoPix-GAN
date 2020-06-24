from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import Activation, LeakyReLU, Concatenate, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from numpy import load, zeros, ones
from numpy.random import randint
from matplotlib import pyplot
import time

start_time = time.time()

def encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(layer_in)

    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2),
                        padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)

    if dropout:
        g = Dropout(0.5)(g, training=True)

    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g


def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)

    input_image_shape = Input(shape=image_shape)

    target_image_shape = Input(shape=image_shape)

    merged = Concatenate()([input_image_shape, target_image_shape])

    # conv block-1
    dis_model = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    dis_model = LeakyReLU(alpha=0.2)(dis_model)

    # conv block-2
    dis_model = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(dis_model)
    dis_model = BatchNormalization()(dis_model)
    dis_model = LeakyReLU(alpha=0.2)(dis_model)

    # conv block-3
    dis_model = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(dis_model)
    dis_model = BatchNormalization()(dis_model)
    dis_model = LeakyReLU(alpha=0.2)(dis_model)

    # conv block-4
    dis_model = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(dis_model)
    dis_model = BatchNormalization()(dis_model)
    dis_model = LeakyReLU(alpha=0.2)(dis_model)

    # conv block-5
    dis_model = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(dis_model)
    dis_model = BatchNormalization()(dis_model)
    dis_model = LeakyReLU(alpha=0.2)(dis_model)

    # conv block-6
    dis_model = Conv2D(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(dis_model)
    patch_out = Activation('sigmoid')(dis_model)

    model = Model(([input_image_shape, target_image_shape]), patch_out)

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    model.summary()

    return model


def define_generator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)
    input_image = Input(shape=image_shape)

    # encoder model
    e1 = encoder_block(input_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)
    # bottleneck , no batch norm and relu
    bottleneck = Conv2D(512, (4, 4), strides=(2, 2),
                        padding='same', kernel_initializer=init)(e7)
    bottleneck = Activation('relu')(bottleneck)

    # decoder model
    d1 = decoder_block(bottleneck, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2),
                        padding='same', kernel_initializer=init)(d7)

    output_image = Activation('tanh')(g)
    model = Model(input_image, output_image)
    model.summary()

    return model


def gan(gen_model, dis_model, image_shape):
    dis_model.trainable = False

    input_src_image = Input(shape=image_shape)

    gen_output_image = gen_model(input_src_image)
    # fed source input image and generator output
    # to discriminator model
    dis_output_image = dis_model([input_src_image, gen_output_image])

    model = Model(input_src_image, [dis_output_image, gen_output_image])

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def load_dataset(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']

    X1 = X1[:500]
    X2 = X2[:500]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return X1, X2


def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    index = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[index], trainB[index]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(gen_model, samples, patch_shape):
    X = gen_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'data/'+'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print(' > Saved: % s and % s' % (filename1, filename2))


# train pix2pix model
def train(dis_model, gen_model, gan_model, dataset, n_epochs=30, n_batch=1):
    n_patch = dis_model.output_shape[1]
    trainA, trainB = dataset
    batch_per_epoch = int(len(trainA) / n_batch)
    n_steps = batch_per_epoch * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(gen_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = dis_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for real samples
        d_loss2 = dis_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d/%d ,d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1 , n_steps, d_loss1, d_loss2, g_loss))

        if (i + 1) % (batch_per_epoch * 10) == 0:
            summarize_performance(i, gen_model, dataset)


# load image data
dataset = load_dataset('data/maps_256.npz')

print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)

print('-----%s seconds-----'%(time.time() - start_time))
