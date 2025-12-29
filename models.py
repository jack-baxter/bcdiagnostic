import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

#build transfer learning model with frozen resnet50 base
#adds global pooling, dense layer, dropout for binary classification
def build_model(config: dict) -> Model:
    target_size = config['target_size']
    
    #load pretrained resnet50 without top classifier
    base_model = ResNet50(
        weights=config['resnet_weights'],
        include_top=False,
        input_shape=(target_size, target_size, 3)
    )
    
    #freeze base layers to use as feature extractor
    base_model.trainable = config['freeze_base']
    
    #add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(config['dense_units'], activation='relu')(x)
    x = Dropout(config['dropout_rate'])(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

#compile model with adam optimizer and binary crossentropy
def compile_model(model: Model, learning_rate: float = 0.001) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#extract feature embeddings from layer before final classification
#used for downstream random forest classifier
def extract_embeddings(model: Model, images: np.ndarray) -> np.ndarray:
    #get output from 3rd layer from end (before dropout and final dense)
    intermediate_model = Model(
        inputs=model.input,
        outputs=model.get_layer(index=-3).output
    )
    
    embeddings = intermediate_model.predict(images, verbose=0)
    
    return embeddings

#save model to disk in keras format
def save_model(model: Model, path: str):
    model.save(path)
    print(f"model saved to {path}")

#load model from disk
def load_saved_model(path: str) -> Model:
    return tf.keras.models.load_model(path)
