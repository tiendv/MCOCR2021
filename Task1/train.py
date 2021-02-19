from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import models
import dataset
import numpy as np
import argparse
import locale
import os
import math
import config

def save_trainging_plot(history, output_filename):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(output_filename)

def train():
    print("[INFO] Loading and cropping all raw training receipt images...")

    images, quality = dataset.load_data(config.TRAIN_IMAGE_FOLDER_PATH, config.TRAIN_CSV_PATH, config.INPUT_SIZE)

    split = train_test_split(quality, images, test_size=0.25, random_state=42)
    (trainY, validY, trainX, validX) = split

    mc = ModelCheckpoint(config.OUTPUT_MODEL_PATH, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model = models.create_efficientnet(config.INPUT_SIZE, config.INPUT_SIZE, 3, config.MODEL_BASE, config.FIRST_LAYERS_TO_FREEZE)
    opt = Adam(lr=config.LEARNING_RATE, decay=config.LEARNING_RATE / config.EPOCHS)
    model.compile(loss="mean_squared_error", optimizer=opt)
    
    # Print the training config
    print(f"[INFO] Training config: \n\
        - MODEL_BASE: {config.MODEL_BASE}\n\
        - INPUT_SIZE: {str(config.INPUT_SIZE)}x{str(config.INPUT_SIZE)}\n\
        - EPOCHS: {str(config.EPOCHS)}\n\
        - BATCH_SIZE: {str(config.BATCH_SIZE)}\n\
        - LEARNING_RATE: {str(config.LEARNING_RATE)}\n")

    # Train the model
    print("[INFO] Training model...")
    H = model.fit(x=trainX, y=trainY, 
        validation_data=(validX, validY),
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=[mc])

    # Save Figure
    print(f"[INFO] Saving the training firgure...")
    save_trainging_plot(H, config.OUTPUT_TRAINING_FIG_PATH)

    # Load saved model
    model = load_model(config.OUTPUT_MODEL_PATH)

    # make predictions on the testing data
    print("[INFO] Predicting receipt quality on validation data...")
    preds = model.predict(validX)

    rmse = math.sqrt(metrics.mean_squared_error(validY, preds))

    print(f"[INFO] RMSE on validation data: {rmse}")

if __name__ == "__main__":
    train()