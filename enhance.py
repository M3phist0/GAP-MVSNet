from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# Load the image
image_path = 'feature_map_w.png'
image = Image.open(image_path)

# Enhance contrast
enhancer = ImageEnhance.Contrast(image)
enhanced_image = enhancer.enhance(1.3)  # Enhance contrast by a factor of 2

# Display the original and enhanced images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap='viridis')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Enhanced Contrast")
# plt.imshow(enhanced_image, cmap='viridis')
# plt.axis('off')

# plt.show()

# Save the enhanced image
enhanced_image_path = 'contrast_enhanced_image1.png'
enhanced_image.save(enhanced_image_path)

enhanced_image_path
