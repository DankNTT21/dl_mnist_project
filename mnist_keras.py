# mnist_keras.py
# Deep Learning CNN on MNIST — fully fixed and ready to run

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def load_and_prep_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save(model, data, epochs=5):
    (x_train, y_train), (x_test, y_test) = data
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128,
                        validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # Save model in single file format (fixed)
    model.save("mnist_keras_model.keras")

    # Plot training accuracy
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Training accuracy')
    plt.savefig('training_accuracy.png')
    plt.close()
    
    return model

def example_prediction(model, data):
    (_, _), (x_test, y_test) = data
    idx = 0
    img = x_test[idx]
    pred = model.predict(np.expand_dims(img, 0))
    predicted_label = np.argmax(pred[0])
    print(f"Predicted: {predicted_label}, True label: {y_test[idx]}")
    plt.imsave('example_digit.png', img.squeeze(), cmap='gray')

def main():
    data = load_and_prep_data()
    model = build_model()
    model = train_and_save(model, data, epochs=5)
    example_prediction(model, data)
    print("Saved: mnist_keras_model.keras, training_accuracy.png, example_digit.png")

if __name__ == "__main__":
    main()