# import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# def load_model() -> Sequential:
#     model = Sequential()

#     model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(256, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(75, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(TimeDistributed(Flatten()))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

#     model.load_weights(os.path.join('models - checkpoint 96','checkpoint'))
    
#     return model


import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# def load_model() -> Sequential:
#     model = Sequential()

#     model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', name='conv3d_1'))
#     model.add(Activation('relu', name='activation_1'))
#     model.add(MaxPool3D((1, 2, 2), name='maxpool3d_1'))

#     model.add(Conv3D(256, 3, padding='same', name='conv3d_2'))
#     model.add(Activation('relu', name='activation_2'))
#     model.add(MaxPool3D((1, 2, 2), name='maxpool3d_2'))

#     model.add(Conv3D(75, 3, padding='same', name='conv3d_3'))
#     model.add(Activation('relu', name='activation_3'))
#     model.add(MaxPool3D((1, 2, 2), name='maxpool3d_3'))

#     model.add(TimeDistributed(Flatten(), name='timedistributed_flatten'))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True), name='bidirectional_1'))
#     model.add(Dropout(.5, name='dropout_1'))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True), name='bidirectional_2'))
#     model.add(Dropout(.5, name='dropout_2'))

#     model.add(Dense(41, kernel_initializer='he_normal', activation='softmax', name='dense'))

#     model.load_weights(os.path.join('models - checkpoint 96', 'checkpoint'))
    
#     return model

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('models - checkpoint 96', 'checkpoint'))
    
    return model
