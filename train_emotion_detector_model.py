import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization


file_path = 'models\emodet.h5'  # Replace with the correct path

if os.path.exists(file_path):
    print("A trained model already exists.")

else:
    tf.keras.preprocessing.image.load_img('data/fer2013/train/Angry/1003.jpg')
    training_generator = ImageDataGenerator(rescale=1./255,
                                            rotation_range=7,
                                            horizontal_flip=True,
                                            zoom_range=0.2)
    train_dataset = training_generator.flow_from_directory('data/fer2013/train',
                                                            target_size = (48, 48),
                                                            batch_size = 16,
                                                            class_mode = 'categorical',
                                                            shuffle = True)

    test_generator = ImageDataGenerator(rescale=1./255)
    test_dataset = test_generator.flow_from_directory('data/fer2013/validation',
                                                    target_size = (48, 48),
                                                    batch_size = 1,
                                                    class_mode = 'categorical',
                                                    shuffle = False)

    num_detectors = 32
    num_classes = 7
    width, height = 48, 48
    epochs = 70

    network = Sequential()

    network.add(Conv2D(num_detectors, (3,3), activation='relu', padding = 'same', input_shape = (width, height, 3)))
    network.add(BatchNormalization())
    network.add(Conv2D(num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.2))

    network.add(Conv2D(2*num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(Conv2D(2*num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.2))

    network.add(Conv2D(2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(Conv2D(2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.2))

    network.add(Conv2D(2*2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(Conv2D(2*2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.2))

    network.add(Flatten())

    network.add(Dense(2 * num_detectors, activation='relu'))
    network.add(BatchNormalization())
    network.add(Dropout(0.2))

    network.add(Dense(2 * num_detectors, activation='relu'))
    network.add(BatchNormalization())
    network.add(Dropout(0.2))

    network.add(Dense(num_classes, activation='softmax'))

    network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_dataset, epochs=epochs)
    network.save('models\emodet.h5')
    print("Finished training.")
