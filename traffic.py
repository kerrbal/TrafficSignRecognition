import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = np.load("x_array.npy")/255.0, np.load("y_array.npy")
    


    the_path = os.path.join(os.curdir, "array.txt")


    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size = 45)
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    print("asdasd")
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []

    labels = []

    max_width = 0
    max_height = 0
    if len(data_dir.split("-")) == 2:
        category_num = 3
    else:
        category_num = 43

    #each category
    for i in range(category_num):
        print(i, "write_tensor")
        current_dir = os.path.join(os.path.curdir, data_dir, str(i))
        all_image_files = os.listdir(current_dir)
        for image_file in all_image_files:
            image_dir = os.path.join(current_dir, image_file)
            file_array = cv2.imread(image_dir, cv2.IMREAD_COLOR)
            
            #all images are have the 20-150 pixel width and same interval of height, so I determined fixed 50, 50 shape because the majority
            #of the images are in the shape between 20-60

            file_array = cv2.resize(file_array, (50, 50))

            images.append(file_array)
            labels.append(i)




    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(40, (2, 2), activation = "relu", input_shape = (50, 50, 3)),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation = "relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(60, activation = "relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model


if __name__ == "__main__":
    main()
