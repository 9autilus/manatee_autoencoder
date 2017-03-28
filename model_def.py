from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda, UpSampling2D
from keras.optimizers import RMSprop

def plot_model(model):
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')

def compile_network(model):
    rms = RMSprop(lr=0.001, decay=0.1)
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

def create_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_img = Input(shape=input_dim)

    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    model = compile_network(model)

    if 0:
        plot_model(model)

    return model