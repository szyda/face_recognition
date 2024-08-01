from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

class FaceRecognizer:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = build_model(input_shape)

    @staticmethod
    def base_model(input_shape):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))

        return model

