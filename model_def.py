from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda, UpSampling2D
from keras.optimizers import RMSprop
from keras.models import load_model

def plot_model(model):
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')

def compile_network(model):
    rms = RMSprop(lr=0.001, decay=0.1)
    model.compile(loss='mse', optimizer=rms)
    return model

def create_network(input_dim):
    encode_layer_id = 0
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_img = Input(shape=input_dim)

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
    decoded = Convolution2D(1, 3, 3, activation='tanh', border_mode='same')(x)
    # tanh non-linearity is required since input is in range (-1, +1)

    model = Model(input_img, decoded)
    model = compile_network(model)

    for i, l in enumerate(model.layers):
        if l.output == encoded:
            encode_layer_id = i

    print(model.summary())

    if 0:
        plot_model(model)

    return model, encode_layer_id


def get_test_model(model_file, input_dim):
    _, encode_layer_id = create_network(input_dim)
    trained_model = load_model(model_file)

    m = Model(trained_model.layers[0].input, trained_model.layers[encode_layer_id].output)
    return m

def get_trained_model(model_file):
    m = load_model(model_file)
