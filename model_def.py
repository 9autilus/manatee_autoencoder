from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda, UpSampling2D
from keras.optimizers import RMSprop
from keras.models import load_model

def plot_model(model):
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')

def compile_network(model, use_binary_sketches):
    rms = RMSprop(lr=0.001, decay=0.1)
    if use_binary_sketches:
        model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    else:
        model.compile(loss='mse', optimizer=rms)
    return model

def create_network(input_dim, use_binary_sketches):
    encode_layer_id = 0
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_img = Input(shape=input_dim)

    if 0:
        x = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        encoded = x
        x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
    elif 0:
        x = Convolution2D(16, 3, 3, activation='tanh', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='tanh', border_mode='same')(x)
        # x = MaxPooling2D((2, 2), border_mode='same')(x)
        # x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        # x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
        # x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='tanh', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
    else:
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        # x = MaxPooling2D((2, 2), border_mode='same')(x)
        # x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        # x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
        # x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)

    if use_binary_sketches:
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)        
    else:    
        decoded = Convolution2D(1, 3, 3, activation='tanh', border_mode='same')(x)

    if 0:
        decoded = Flatten()(decoded)

    # tanh non-linearity is required since input is in range (-1, +1)

    model = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    model = compile_network(model, use_binary_sketches)
    encoder = compile_network(encoder, use_binary_sketches)

    if 0:
        plot_model(model)

    return model, encoder


def load_half_model(model_file, input_dim):
    m_fresh, encoder, encode_layer_id = create_network(input_dim)
    trained_model = load_model(model_file)

    if 0:
        m = Model(trained_model.layers[0].input, trained_model.layers[encode_layer_id].output)
    else:
        x = Flatten()(trained_model.layers[encode_layer_id].output)
        m = Model(trained_model.layers[0].input, x)
    # m = Model(m_fresh.layers[0].input, m_fresh.layers[encode_layer_id].output)

    m.save('test2.h5')

    for lyr in m.layers:
        lyr.trainable = False

    return m

def load_autoencoder(model_file):
    return load_model(model_file.split('.')[0] + '_ae.' + model_file.split('.')[1])

def load_encoder(model_file, use_binary_sketches):
    m = load_model(model_file.split('.')[0] + '_e.' + model_file.split('.')[1])
    m = compile_network(m, use_binary_sketches)

    return m
