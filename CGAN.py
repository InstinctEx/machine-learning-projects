import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

#For my 3060 laptop gpu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

policy = mixed_precision.Policy('mixed_float16')  
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)  # xla gia optimize

# random number stabilization
tf.random.set_seed(42)
np.random.seed(42)

# parametroi ekpaideushs
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64, 64, 3
BATCH_SIZE = 512  
LATENT_DIM = 100  # noise for gen
NUM_CLASSES = 5   # poses hlikies exoume
EPOCHS = 1000     

# epeksergasia ikonas me d.a. kai kathgoropoihsh hlikias
def load_and_process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.random_flip_left_right(img)
    img = (img / 127.5) - 1.0  # kanonikopoihsh sto [-1, 1]

    filename = tf.strings.split(file_path, os.sep)[-1]
    age_str = tf.strings.split(filename, "_")[0]
    age = tf.strings.to_number(age_str, out_type=tf.int32)

    conditions = [
        age <= 20,  
        age <= 35,  
        age <= 55, 
        age <= 65, 
        age > 65   
    ]
    age_group = tf.argmax(tf.stack(conditions, axis=0), axis=0)

    return img, age_group

# Φόρτωση UTKFace
data_dir = os.path.join("UTKFace")
file_pattern = os.path.join(data_dir, "*.jpg")
all_files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
total_files = len(list(all_files))
train_size = int(0.8 * total_files)  # 80% tou dataset gia train
train_files = all_files.take(train_size)
val_files = all_files.skip(train_size)  # 20% gia validate

# Προετοιμασία dataset με cache για ταχύτητα
train_data = train_files.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(6000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_data = val_files.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Δημιουργία Generator με περισσότερα layers
def create_generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    labels = layers.Input(shape=(1,), dtype='int32')
    label_emb = layers.Embedding(NUM_CLASSES, 50)(labels)
    label_flat = layers.Flatten()(label_emb)
    combined = layers.Concatenate()([noise, label_flat])
    x = layers.Dense(4 * 4 * 512, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, 4, strides=1, padding='same', use_bias=False, activation='tanh')(x)
    return Model([noise, labels], x, name="generator")

# Δημιουργία Discriminator με Dropout
def create_discriminator():
    img_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_emb = layers.Embedding(NUM_CLASSES, IMG_HEIGHT * IMG_WIDTH)(label_input)
    label_reshaped = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(layers.Flatten()(label_emb))
    x = layers.Concatenate(axis=-1)([img_input, label_reshaped])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return Model([img_input, label_input], x, name="discriminator")

# Απώλειες
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def disc_loss_fn(real_logits, fake_logits):
    real_loss = cross_entropy_loss(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy_loss(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss

def gen_loss_fn(fake_logits):
    return cross_entropy_loss(tf.ones_like(fake_logits), fake_logits)

# Μοντέλα και optimizers
gen_model = create_generator()
disc_model = create_discriminator()
gen_opt = Adam(learning_rate=2e-4, beta_1=0.5)
disc_opt = Adam(learning_rate=2e-4, beta_1=0.5)

# Φόρτωση μοντέλων αν υπάρχουν
if os.path.exists('gen_best.keras'):
    gen_model = tf.keras.models.load_model('gen_best.keras')
    print("Loaded saved generator model")
if os.path.exists('disc_best Ascendancy:'):
    disc_model = tf.keras.models.load_model('disc_best.keras')
    print("Loaded saved discriminator model")

# Βήμα εκπαίδευσης
@tf.function
def training_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    random_labels = tf.random.uniform([BATCH_SIZE, 1], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = gen_model([noise, random_labels], training=True)
        real_labels_exp = tf.expand_dims(labels, axis=-1)
        real_out = disc_model([images, real_labels_exp], training=True)
        fake_out = disc_model([fake_images, random_labels], training=True)
        g_loss = gen_loss_fn(fake_out)
        d_loss = disc_loss_fn(real_out, fake_out)
    gen_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
    disc_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, gen_model.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, disc_model.trainable_variables))
    return g_loss, d_loss

# Validation
def validate_model(dataset):
    total_loss = 0
    batches = 0
    for imgs, lbls in dataset:
        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        rand_lbls = tf.random.uniform([BATCH_SIZE, 1], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
        fake_imgs = gen_model([noise, rand_lbls], training=False)
        real_lbls_exp = tf.expand_dims(lbls, axis=-1)
        real_out = disc_model([imgs, real_lbls_exp], training=False)
        fake_out = disc_model([fake_imgs, rand_lbls], training=False)
        d_loss = disc_loss_fn(real_out, fake_out)
        total_loss += d_loss
        batches += 1
    return total_loss / batches

# Παραγωγή εικόνων
def create_and_save_images(model, epoch):
    noise_input = tf.random.normal([NUM_CLASSES, LATENT_DIM])
    label_input = tf.convert_to_tensor([[i] for i in range(NUM_CLASSES)], dtype=tf.int32)
    preds = model([noise_input, label_input], training=False)
    preds = (preds + 1) / 2.0  # Μετατροπή από [-1, 1] σε [0, 1]
    preds = tf.cast(preds, tf.float32).numpy()
    fig = plt.figure(figsize=(NUM_CLASSES, 1))
    for i in range(NUM_CLASSES):
        plt.subplot(1, NUM_CLASSES, i + 1)
        plt.imshow(preds[i])
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()

# Εκπαίδευση
def train_network(train_ds, val_ds, epochs):
    best_val_loss = float('inf')
    patience = 100  # Αυξημένο patience για αποφυγή πρόωρου early stopping
    wait = 0
    gen_losses = []
    disc_losses = []
    val_losses = []

    for ep in range(epochs):
        print(f"Epoch {ep + 1}/{epochs}")
        for img_batch, lbl_batch in train_ds:
            g_loss, d_loss = training_step(img_batch, lbl_batch)
        val_loss = validate_model(val_ds)
        gen_losses.append(g_loss)
        disc_losses.append(d_loss)
        val_losses.append(val_loss)
        print(f"Gen Loss: {g_loss:.4f} | Disc Loss: {d_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            gen_model.save('gen_best.keras')
            disc_model.save('disc_best.keras')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {ep + 1}")
                break
        if (ep + 1) % 10 == 0:
            create_and_save_images(gen_model, ep + 1)

# Εκκίνηση εκπαίδευσης
train_network(train_data, val_data, EPOCHS)
