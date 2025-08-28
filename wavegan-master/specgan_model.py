import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# Define the SpecGAN Generator
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(16 * 16 * 256, input_dim=latent_dim),
        layers.Reshape((16, 16, 256)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")
    ])
    return model

# Define the SpecGAN Discriminator
def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Define the GAN class
class SpecGAN:
    def __init__(self, latent_dim, input_shape):
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator(input_shape)
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])

        self.gan = self._build_gan()

    def _build_gan(self):
        self.discriminator.trainable = False
        model = tf.keras.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
        return model

    def train(self, data, epochs, batch_size):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_data = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_data, real)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real)

            print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f} | G Loss: {g_loss:.4f}")

# Load the dataset
def load_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(data_dir, file)
            spectrogram = np.load(file_path)
            data.append(spectrogram)
    return np.expand_dims(np.array(data), axis=-1)

if __name__ == "__main__":
    # Parameters
    latent_dim = 100
    input_shape = (128, 128, 1)  # Assuming spectrograms are 128x128
    data_dir = r"C:\Users\domna735\OneDrive\Desktop\Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration\processed_data"

    # Load data
    data = load_data(data_dir)

    # Initialize and train SpecGAN
    specgan = SpecGAN(latent_dim, input_shape)
    specgan.train(data, epochs=1000, batch_size=32)
