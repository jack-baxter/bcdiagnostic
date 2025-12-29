import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

#train model with validation data and optional checkpointing
def train_model(model: Model, 
               train_images: np.ndarray, 
               train_labels: np.ndarray,
               test_images: np.ndarray, 
               test_labels: np.ndarray,
               config: dict,
               checkpoint_path: str = None) -> dict:
    
    callbacks = []
    
    #save best model based on val accuracy
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    #train with validation split
    history = model.fit(
        train_images, 
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks if callbacks else None,
        verbose=1
    )
    
    return history.history

#evaluate model on test set
def evaluate_model(model: Model, 
                  test_images: np.ndarray, 
                  test_labels: np.ndarray) -> dict:
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy
    }

#get predictions from model
def predict(model: Model, images: np.ndarray) -> np.ndarray:
    predictions = model.predict(images, verbose=0)
    return (predictions > 0.5).astype(int).flatten()
