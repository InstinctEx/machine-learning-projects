# Import required libraries
import numpy as np                  # Numerical computing and array operations
import pandas as pd                 # Data manipulation and analysis
import tensorflow as tf             # Deep learning framework
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # Pretrained CNN model
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Image processing
from tensorflow.keras.preprocessing.text import Tokenizer               # Text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences       # Sequence padding
from tensorflow.keras.models import Model                              # Neural network model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, RepeatVector, concatenate  # NN layers
from tensorflow.keras.callbacks import ModelCheckpoint                 # Model checkpointing
import os                               # Operating system operations
from PIL import Image                   # Image handling


DATASET_DIR = 'dataset/'                # Root directory containing dataset
IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')          # Path to images folder
CAPTIONS_FILE = os.path.join(DATASET_DIR, 'captions.txt') # Path to captions file

# Model architecture parameters
MAX_CAPTION_LENGTH = 40     # Maximum length of captions (truncate longer captions)
VOCAB_SIZE = 20000          # Maximum size of vocabulary (top n frequent words)
EMBEDDING_DIM = 256         # Dimension of word embedding vectors
LSTM_UNITS = 512            # Number of units in LSTM layer
DENSE_UNITS = 512           # Number of units in dense layer

# Training parameters
BATCH_SIZE = 64           # Number of samples per training batch
EPOCHS = 50                 # Number of training epochs
# Load and preprocess captions
df = pd.read_csv(CAPTIONS_FILE)  # Read the captions file into a DataFrame
captions = df['caption'].tolist()  # Convert the 'caption' column into a list of captions

# Limit dataset to 5 samples for quick testing
df = df.head(5)  # Only keep the first 5 samples for quick testing

def preprocess_text(text):
    # Preprocess text by converting to lowercase and removing special characters
    text = text.lower()
    text = text.replace('[^A-Za-z0-9 ]+', '')  # Remove special characters
    return 'startseq ' + text + ' endseq'  # Add 'startseq' at the beginning and 'endseq' at the end

# Apply preprocessing to each caption
df['processed_caption'] = df['caption'].apply(preprocess_text)

# Create a tokenizer to convert words into integer tokens
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<unk>')  # OOV (out of vocabulary) token for unknown words
tokenizer.fit_on_texts(df['processed_caption'])  # Fit the tokenizer on the processed captions
VOCAB_SIZE = len(tokenizer.word_index) + 1  # Update VOCAB_SIZE based on the tokenizer's word index

# Convert captions to sequences of integers
sequences = tokenizer.texts_to_sequences(df['processed_caption'])
# Pad sequences to ensure they all have the same length
padded_sequences = pad_sequences(sequences, maxlen=MAX_CAPTION_LENGTH, padding='post')

# Split the data into inputs (X_seq) and targets (y)
X_seq = padded_sequences[:, :-1]  # Input sequence (without the last word)
y = padded_sequences[:, 1:]  # Target sequence (without the first word)
# Create sample weights to ignore padding tokens during training
sample_weights = np.where(y != 0, 1, 0).astype(np.float32)

# Load and preprocess images
def extract_features(image_path):
    # Load a pre-trained VGG16 model to extract features from images
    model = VGG16(include_top=False, weights='imagenet', pooling='avg')  # Exclude the top classification layers
    # Load the image and resize it to 224x224 (required by VGG16)
    image = load_img(image_path, target_size=(224, 224))
    # Convert the image to an array
    image = img_to_array(image)
    # Preprocess the image for VGG16
    image = preprocess_input(image)
    # Get the features from the image by passing it through the VGG16 model
    features = model.predict(np.expand_dims(image, axis=0))
    return features[0]  # Return the features (1D array)

# Precompute features for each image
def precompute_image_features(df, images_dir):
    image_features = {}
    # For each unique image in the DataFrame, extract its features
    for img_file in df['image'].unique():
        img_path = os.path.join(images_dir, img_file)  # Get the full path to the image
        image_features[img_file] = extract_features(img_path)  # Extract image features and store them
    return image_features

# Load image features for the 5 images in the dataset
image_features = precompute_image_features(df, IMAGES_DIR)

# Prepare image data (convert image features into a numpy array)
X_img = np.array([image_features[img] for img in df['image']])

# Split the dataset into training and validation sets
X_img_train, X_img_val = X_img[:4], X_img[4:]  #  4 images gia training, teleutaia gia validation
X_seq_train, X_seq_val = X_seq[:4], X_seq[4:]  #  4 sequences gia training, teleutaia gia validation
y_train, y_val = y[:4], y[4:]  # 4 sequences gia training, teleutaia gia validation
sample_weights_train, sample_weights_val = sample_weights[:4], sample_weights[4:]

# Build the model
# Input layer for image features (512-dimensional vector)
image_input = Input(shape=(512,))
image_dense = Dense(DENSE_UNITS, activation='relu')(image_input)  # Dense layer to process image features
image_rep = RepeatVector(MAX_CAPTION_LENGTH-1)(image_dense)  # Repeat image features to match caption length

# Input layer for caption sequences (max length of the caption)
caption_input = Input(shape=(MAX_CAPTION_LENGTH-1,))
caption_emb = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=False)(caption_input)  # Embedding layer for captions

# Merge image and caption 
merged = concatenate([image_rep, caption_emb])

# LSTM layer to predict the next word in the caption
lstm_out = LSTM(LSTM_UNITS, return_sequences=True)(merged)
# Output layer to predict the next word in the caption
outputs = Dense(VOCAB_SIZE, activation='softmax')(lstm_out)

# Define the model
model = Model(inputs=[image_input, caption_input], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')  # Compile the model with loss and optimizer

# save the best model based on validation loss
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    [X_img_train, X_seq_train], y_train,  # Training data 
    batch_size=BATCH_SIZE,  
    epochs=EPOCHS,  
    validation_data=([X_img_val, X_seq_val], y_val, sample_weights_val), 
    callbacks=[checkpoint],  # save the best model
    sample_weight=sample_weights_train,
    verbose=1  # Show training progress
)

# generating captions
def generate_caption_with_temperature(model, image_file, tokenizer, max_length=MAX_CAPTION_LENGTH, temperature=1.0):
    # Get the features for the input image
    image_feature = image_features[image_file].reshape(1, -1)  # Ensure shape is (1, 512)
    caption = 'startseq'  # Start the caption with the 'startseq' token
    
    # Generate the caption word by word
    for _ in range(max_length - 1):
        sequence = tokenizer.texts_to_sequences([caption])[0]  # Convert the caption to a sequence of tokens
        sequence = pad_sequences([sequence], maxlen=max_length - 1)  # Pad the sequence to match max length
        
        # Make prediction using the model (get probability distribution over next word)
        pred = model.predict([image_feature, sequence], verbose=0)
        
        # Temperature sampling for creativity in predictions
        pred = pred[0, -1, :]  # Get the probabilities for the next word
        pred = np.exp(pred / temperature)  # Apply temperature scaling
        pred = pred / np.sum(pred)  # Normalize to get probabilities

        # Sample the next word based on the probabilities
        pred_word = np.random.choice(len(pred), p=pred)
        next_word = tokenizer.index_word.get(pred_word, '')  # Get the word from the index
        
        # Stop if 'endseq' is predicted
        if next_word == 'endseq':
            break
        
        caption += ' ' + next_word  # Append the next word to the caption
    
    return caption.replace('startseq ', '').replace(' endseq', '')  # Remove 'startseq' and 'endseq'

# Example usage: Generate a caption for the test image (the last image in the dataset)
test_image = df['image'].iloc[4]  # Get the last image for testing
print(generate_caption_with_temperature(model, test_image, tokenizer))  # Generate and print the caption