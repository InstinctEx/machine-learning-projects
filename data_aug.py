# - CIFAR-10 (Εικόνες): https://www.cs.toronto.edu/~kriz/cifar.html
# - Daily Minimum Temperatures in Melbourne (Χρονοσειρές):
https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-mintemperatures.csv
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Ρύθμιση του backend σε Agg για αποθήκευση χωρίς
εμφάνιση
import matplotlib.pyplot as plt
# load cifar 10
def load_cifar10_image():
"""Φορτώνει μία εικόνα από το CIFAR-10 dataset."""
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
return x_train[0] # 1st image
def augment_image(image):
augmented_images = []
# 1. Γεωμετρικοί Μετασχηματισμοί
# Περιστροφή
angle = np.random.uniform(-30, 30)
h, w = image.shape[:2]
M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
rotated = cv2.warpAffine(image, M, (w, h))
augmented_images.append(("Rotation", rotated))
# Μετατόπιση
tx, ty = np.random.randint(-5, 6, 2)
M = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(image, M, (w, h))
augmented_images.append(("Translation", translated))
# Κλιμάκωση
scaled = cv2.resize(image, None, fx=1.2, fy=1.2,
interpolation=cv2.INTER_LINEAR)
scaled = cv2.resize(scaled, (w, h)) # Επαναφορά στο αρχικό μέγεθος
augmented_images.append(("Scaling", scaled))
# Κάτοπτρο (Οριζόντιο)
flipped = cv2.flip(image, 1)
augmented_images.append(("Flipping", flipped))
# Κοπή
crop_size = 28
x, y = np.random.randint(0, h - crop_size, 2)
cropped = image[y:y+crop_size, x:x+crop_size]
cropped = cv2.resize(cropped, (h, w))
augmented_images.append(("Cropping", cropped))
# 2. Αλλαγές στις Χρωματικές Αποχρώσεις (με PIL)
pil_img = Image.fromarray(image)
# Φωτεινότητα
enhancer = ImageEnhance.Brightness(pil_img)
bright = enhancer.enhance(np.random.uniform(0.5, 1.5))
augmented_images.append(("Brightness", np.array(bright)))
# Αντίθεση
enhancer = ImageEnhance.Contrast(pil_img)
contrast = enhancer.enhance(np.random.uniform(0.5, 1.5))
augmented_images.append(("Contrast", np.array(contrast)))
# Κορεσμός
enhancer = ImageEnhance.Color(pil_img)
saturation = enhancer.enhance(np.random.uniform(0.5, 1.5))
augmented_images.append(("Saturation", np.array(saturation)))
return augmented_images
# --- Επαύξηση Χρονοσειρών ---
def load_timeseries():
"""ελάχιστες ημερήσιες θερμοκρασίες"""
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/dailymin-temperatures.csv"
df = pd.read_csv(url)
return df["Temp"].values # Επιστρέφει τις θερμοκρασίες ως πίνακα
def augment_timeseries(series):
"""τεχνικές σε μία χρονοσειρά."""
augmented_series = []
# 1. Προσθήκη Θορύβου
noise = np.random.normal(0, 0.5, series.shape)
noisy_series = series + noise
augmented_series.append(("Noise", noisy_series))
# 2. Παράθυρα Χρόνου
window_size = int(len(series) * 0.8)
start = np.random.randint(0, len(series) - window_size)
windowed = series[start:start + window_size]
augmented_series.append(("Window Slicing", windowed))
# 3. Μετατόπιση Χρόνου
shift = np.random.randint(-10, 10)
shifted = np.roll(series, shift)
augmented_series.append(("Time Shifting", shifted))
# 4. Διακύμανση
scale_factor = np.random.uniform(0.8, 1.2)
scaled = series * scale_factor
augmented_series.append(("Scaling", scaled))
# 5. Αντιστροφή
reversed_series = series[::-1]
augmented_series.append(("Reversing", reversed_series))
# 6. Παρεμβολή
x = np.arange(len(series))
x_new = np.linspace(0, len(series) - 1, len(series) + 10) # Προσθήκη 10
σημείων
interpolated = np.interp(x_new, x, series)
augmented_series.append(("Interpolation", interpolated))
return augmented_series
# Αποθήκευση Γραφημάτων
# Εικόνες
original_image = load_cifar10_image()
augmented_images = augment_image(original_image)
# Δημιουργία φιγούρας για εικόνες
plt.figure(figsize=(15, 5))
plt.subplot(2, 5, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
for i, (title, img) in enumerate(augmented_images[:9], 2): # Περιορισμός στα
9
plt.subplot(2, 5, i)
plt.imshow(img)
plt.title(title)
plt.axis("off")
plt.tight_layout()
plt.savefig('augmented_images.png') # Αποθήκευση
# Χρονοσειρές
original_series = load_timeseries()
augmented_series = augment_timeseries(original_series)
# Δημιουργία φιγούρας για χρονοσειρές
plt.figure(figsize=(15, 12))
plt.subplot(4, 2, 1)
plt.plot(original_series)
plt.title("Original Timeseries")
plt.xlabel("Time (Days)")
plt.ylabel("Temperature (°C)")
for i, (title, series) in enumerate(augmented_series, 2):
plt.subplot(4, 2, i)
plt.plot(series)
plt.title(title)
plt.xlabel("Time (Days)")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig('augmented_timeseries.png')
