model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(40, (2, 2), activation = "relu", input_shape = (50, 50, 3)),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation = "relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(30, activation = "relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])

epoch number = 15, batch size = 45

In these configurations, my model achieved a 96% accuracy rate, which is quite high. I tested it with batch sizes of 45 and 15. Training was significantly faster with batch size 45. This is because GPUs efficiently handle larger batch sizes through parallel processing. Although each step processes more samples, fewer steps are required per epoch, reducing total training time.

Accuracy remained similar between batch sizes, but I noticed greater fluctuation when using batch size 15. This happens because smaller batches lead to more frequent weight updates, making the model more sensitive to noise in the dataset. As a result, the training process reacts more to little variations, causing instability in accuracy. On the other hand, a larger batch size smooths out these variations, leading to a more stable training process.

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(40, (2, 2), activation = "relu", input_shape = (50, 50, 3)),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation = "relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(30, activation = "relu"),
        tf.keras.layers.Dense(5, activation = "relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])

with that configuration, it gave 90% accuracy rate. it decreassed comparing to the previous one. I think it is a result of last hidden layer.
Because it only has 5 neurons and we have 43 categories. 5 Neurons cannot realize much attributes about the images for categorizing it into 43 categories.
It is called bottleneck effect.

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(40, (2, 2), activation = "relu", input_shape = (50, 50, 3)),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation = "relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(60, activation = "relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])
this neural network yields 97 percent accuracy rate. The network extracts more meaningful features from the image. The additional neurons in dense layers help process and retain these features better.

