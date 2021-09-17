"""
This is my first VAE implementation.
It encodes the MNIST dataset into a 16-dimensional space.
Most of the code was copied from: https://keras.io/getting_started/intro_to_keras_for_researchers/
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]  # dimensionality of the latent space
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # re-parametrization trick


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampler()

    def call(self, inputs, **kwargs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs, **kwargs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VAE(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var, z = self.encoder(inputs, **kwargs)
        reconstructed = self.decoder(z, **kwargs)

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

    def generate_number(self):
        z = tf.keras.backend.random_normal(shape=(1, self.latent_dim))
        return self.decoder(z)


if __name__ == "__main__":
    # Our model.
    vae = VAE(original_dim=784, intermediate_dim=64, latent_dim=16)

    # Loss and optimizer.
    loss_fn = tf.keras.losses.MeanSquaredError()  # Compare pixel to pixel
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Prepare a dataset.
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        x_train.reshape(60000, 784).astype("float32") / 255
    )
    dataset = dataset.shuffle(buffer_size=1024).batch(32)


    @tf.function
    def training_step(x):
        with tf.GradientTape() as tape:
            reconstructed = vae(x)  # Compute input reconstruction.
            # Compute loss.
            loss = loss_fn(x, reconstructed)
            loss += sum(vae.losses)  # Add KLD term.
        # Update the weights of the VAE.
        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        return loss


    losses = []  # Keep track of the losses over time.
    for epoch in range(2):
        for step, x in enumerate(dataset):
            loss = training_step(x)
            # Logging.
            losses.append(float(loss))
            if step % 100 == 0:
                print("Epoch:", epoch, "Step:", step, "Loss:", sum(losses) / len(losses))



