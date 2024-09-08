import cv2
import matplotlib.pyplot as plt


# Load the first image
image1 = cv2.imread("image_1.jpg")

# Convert to grayscale
img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply thresholding
ret, thresh1 = cv2.threshold(img_gray1, 150, 255, cv2.THRESH_BINARY)

# Display the image
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(img_gray1, cmap='gray')
plt.title('Grayscale Image')
plt.subplot(1, 3, 3)
plt.imshow(thresh1, cmap='gray')
plt.title('Thresholded Image')
plt.tight_layout()
plt.show()

# Detect the contours on binary image
contours1, hierarchy1 = cv2.findContours(image=thresh1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# Draw contours on original image
image_contours1 = cv2.drawContours(image=image1.copy(), contours=contours1, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

# Display image with contours
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image_contours1, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
plt.title('Contours')
plt.show()

# Repeat the process for the second image
image2 = cv2.imread("image_2.jpg")
img_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(img_gray2, 150, 255, cv2.THRESH_BINARY)

# Display the image
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(img_gray2, cmap='gray')
plt.title("Grayscale Image")
plt.subplot(1, 3, 3)
plt.imshow(thresh2, cmap='gray')
plt.title('Thresholded Image')
plt.tight_layout()
plt.show()

# Process with different contour retrieval modes
modes = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE]
plt.figure(figsize=(10, 10))
for i, mode in enumerate(modes):
    contours2, hierarchy2 = cv2.findContours(image=thresh2, mode=mode, method=cv2.CHAIN_APPROX_NONE)
    image_contours2 = cv2.drawContours(image=image2.copy(), contours=contours2, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    plt.subplot(2, 2, i + 1)
    plt.imshow(cv2.cvtColor(image_contours2, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
    plt.title(str(mode))
    print(f'str({mode}): {hierarchy2}')
plt.tight_layout()
plt.show()

# Process the third image
image3 = cv2.imread("custom_colors.jpg")
img_gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
ret, thresh3 = cv2.threshold(img_gray3, 150, 255, cv2.THRESH_BINARY)

# Display the image
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(img_gray3, cmap='gray')
plt.title('Grayscale Image')
plt.subplot(1, 3, 3)
plt.imshow(thresh3, cmap='gray')
plt.title('Thresholded Image')
plt.tight_layout()
plt.show()

# Process with different contour approximation methods
methods = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE]
plt.figure(figsize=(12, 8))
for i, method in enumerate(methods):
    contours3, hierarchy3 = cv2.findContours(image=thresh3, mode=cv2.RETR_TREE, method=method)
    image_contours3 = cv2.drawContours(image=image3.copy(), contours=contours3, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    plt.subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(image_contours3, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
    plt.title(str(method))
plt.tight_layout()
plt.show()

# Draw circles on contour points for different approximation methods
methods = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE]
plt.figure(figsize=(12, 8))
for l, method in enumerate(methods):
    contours3, hierarchy3 = cv2.findContours(image=thresh3, mode=cv2.RETR_TREE, method=method)
    image_contours3 = image3.copy()
    for i, contour in enumerate(contours3):  # Loop over each contour
        for j, point in enumerate(contour):  # Loop over each point in the contour
            # Draw a circle on the current contour coordinate
            cv2.circle(image_contours3, (point[0][0], point[0][1]), 2, (255, 0, 0), 2, cv2.LINE_AA)
    plt.subplot(1, 2, l + 1)
    plt.imshow(cv2.cvtColor(image_contours3, cv2.COLOR_BGR2RGB))
    plt.title(str(method))
plt.tight_layout()
plt.show()