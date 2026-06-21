"""
EE200 Practical Assignment - Image Frequency Analysis and Fusion
Name: Aaditya Singh
Roll Number: 240008
Topic: 2D Discrete Fourier Transform, Spectrum Analysis, Image Fusion
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Loading Images 
cat = Image.open("C:\\Users\\aadit\\OneDrive\\Pictures\\cat_gray.jpg").convert('L').resize((512, 512))
dog = Image.open("C:\\Users\\aadit\\OneDrive\\Pictures\\dog_gray.jpg").convert('L').resize((512, 512))

# Frequency Analysis on Cat
cat_array = np.array(cat)
F = np.fft.fft2(cat_array)
magnitude = np.abs(F)
magnitude_db = 20 * np.log10(magnitude + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(magnitude, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_db, cmap='gray')
plt.title("Magnitude Spectrum (dB)")
plt.axis('off')
plt.tight_layout()
plt.show()

# STEP 3: Centered Spectrum
F_shifted = np.fft.fftshift(F)
magnitude_shifted_db = 20 * np.log10(np.abs(F_shifted) + 1)

plt.imshow(magnitude_shifted_db, cmap='gray')
plt.title("Centered Magnitude Spectrum (dB)")
plt.axis('off')
plt.show()

# STEP 4: Rotation Effect
cat_rotated = cat.rotate(90)
cat_rotated_array = np.array(cat_rotated)
F_rotated = np.fft.fftshift(np.fft.fft2(cat_rotated_array))
magnitude_rotated_db = 20 * np.log10(np.abs(F_rotated) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(magnitude_shifted_db, cmap='gray')
plt.title("Original Spectrum (Centered, dB)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_rotated_db, cmap='gray')
plt.title("Rotated Image Spectrum (Centered, dB)")
plt.axis('off')
plt.tight_layout()
plt.show()

# STEP 5: Frequency Domain Fusion 
cat_np = np.array(cat.resize((256, 256)))
dog_np = np.array(dog.resize((256, 256)))

F1 = np.fft.fftshift(np.fft.fft2(dog_np))  # low-freq from dog
F2 = np.fft.fftshift(np.fft.fft2(cat_np))  # high-freq from cat

rows, cols = F1.shape
crow, ccol = rows // 2, cols // 2
radius = 40

Y, X = np.ogrid[:rows, :cols]
distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
mask = distance <= radius

plt.imshow(mask, cmap='gray')
plt.title("Transfer Function (Low-Pass Mask)")
plt.axis('off')
plt.show()

F_fused = F1 * mask + F2 * (~mask)
fused_image = np.abs(np.fft.ifft2(np.fft.ifftshift(F_fused)))

plt.imshow(fused_image, cmap='gray')
plt.title("Fused Image (Frequency Mixer Output)")
plt.axis('off')
plt.show()

# frequency mixing(Dog + Cat)

# Convert to float32
cat_np = np.array(cat, dtype=np.float32)
dog_np = np.array(dog, dtype=np.float32)

# Low-frequency dog: blur
dog_blurred = dog.filter(ImageFilter.GaussianBlur(radius=8))
dog_blurred_np = np.array(dog_blurred, dtype=np.float32)

# High-frequency cat: subtract blurred version
cat_blurred = cat.filter(ImageFilter.GaussianBlur(radius=3))
cat_blurred_np = np.array(cat_blurred, dtype=np.float32)
cat_highpass = cat_np - cat_blurred_np

# Normalize high-pass
cat_highpass -= cat_highpass.min()
cat_highpass /= cat_highpass.max()
cat_highpass *= 255

# Combine to form hybrid
hybrid = dog_blurred_np + cat_highpass
hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(dog_np, cmap='gray')
plt.title("Original Dog")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cat_np, cmap='gray')
plt.title("Original Cat")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(hybrid, cmap='gray')
plt.title("Hybrid Image (Squint to See Cat)")
plt.axis('off')
plt.tight_layout()
plt.show()

# Comparing rotated image spectra with original image's spectra, and writting observations
'''After rotating the image 90 degrees anti-clockwise, the 2D Fourier transform magnitude spectrum also rotates by 90 degrees in the same direction.
The intensity distribution remains unchanged, but the orientation of the frequency components shifts, confirming that the Fourier transform 
preserves the structure of an image in the frequency domain while aligning it to the image's spatial orientation. This property illustrates the
close link between geometric transformations in space and their mirror in the frequency world. '''