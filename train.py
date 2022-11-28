import torch
print(torch.cuda.get_device_name())
from generator import generator
import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy
from discriminator import discriminator
from fashion_GAN import FashionGAN
import tensorflow_datasets as tfds
import time

g_opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 
d_opt = tf.keras.optimizers.Adam(learning_rate=0.00001) 
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()


generator.load_weights("generator_500.h5")
discriminator.load_weights("discriminator_501.h5")

model = FashionGAN(generator,discriminator)

model.compile(g_opt, d_opt, g_loss, d_loss)

def scale_images(data): 
    image = data['image']
    return image / 255
ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_images) 
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(120)
ds = ds.prefetch(64)

for i in range(9):
    hist = model.fit(
        ds, 
        epochs=50, 
        #callbacks=[ModelMonitor()],
        )
    generator.save(f'generator_{50*(i+2)}.h5')
    discriminator.save(f'discriminator_{50*(i+2)}.h5')
    time.sleep(1200)