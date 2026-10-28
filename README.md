# edge-detection-opencv

## Aim

To perform edge detection using Sobel, Roberts, Prewitt, Laplacian, and Canny edge detectors.

---

## Software Required

- Anaconda – Python 3.7  
- Jupyter Notebook / VS Code  
- OpenCV (cv2)  
- NumPy  
- Matplotlib  

---

## ⚙️ Algorithm

### Step 1:
Import all the necessary modules for the program.

### Step 2:
Load an image using `cv2.imread()`.

### Step 3:
Convert the image to grayscale.

### Step 4:
Apply **Sobel operator** using OpenCV to detect edges.

### Step 5:
Apply **Prewitt operator** using custom kernels.

### Step 6:
Apply **Roberts operator** using custom kernels.

### Step 7:
Apply **Laplacian operator** using OpenCV.

### Step 8:
Apply **Canny edge detector** using OpenCV.

### Step 9:
Display all edge-detected images for comparison.

---

## Developed By

- **Name:**Sharmili V 
- **Register No:** 212225040405

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image6.jpg")

if img is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# Sobel X
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Combine both directions
sobel = cv2.magnitude(sobel_x, sobel_y)

sobel = cv2.normalize(
    sobel, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

prewitt_x_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

prewitt_y_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

prewitt_x = cv2.filter2D(
    gray, cv2.CV_32F, prewitt_x_kernel
)

prewitt_y = cv2.filter2D(
    gray, cv2.CV_32F, prewitt_y_kernel
)

prewitt = cv2.magnitude(prewitt_x, prewitt_y)

prewitt = cv2.normalize(
    prewitt, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

roberts_x_kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

roberts_y_kernel = np.array([
    [0, 1],
    [-1, 0]
], dtype=np.float32)

roberts_x = cv2.filter2D(
    gray, cv2.CV_32F, roberts_x_kernel
)

roberts_y = cv2.filter2D(
    gray, cv2.CV_32F, roberts_y_kernel
)

roberts = cv2.magnitude(roberts_x, roberts_y)

roberts = cv2.normalize(
    roberts, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

laplacian = cv2.Laplacian(
    gray, cv2.CV_64F, ksize=3
)

laplacian = cv2.convertScaleAbs(laplacian)
canny = cv2.Canny(gray, 80, 160)

# Convert original image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 9))

# 1. Original
plt.subplot(2, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# 2. Grayscale
plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# 3. Sobel
plt.subplot(2, 4, 3)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Operator")
plt.axis("off")

# 4. Prewitt
plt.subplot(2, 4, 4)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt Operator")
plt.axis("off")

# 5. Roberts
plt.subplot(2, 4, 5)
plt.imshow(roberts, cmap="gray")
plt.title("Roberts Operator")
plt.axis("off")

# 6. Laplacian
plt.subplot(2, 4, 6)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Operator")
plt.axis("off")

# 7. Canny
plt.subplot(2, 4, 7)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detector")
plt.axis("off")

plt.tight_layout()
plt.show()

```
---

## Output
<img width="635" height="350" alt="image" src="https://github.com/user-attachments/assets/d6300323-ef25-42a9-ab6f-1ecb5e8e4be8" />

<img width="1327" height="230" alt="image" src="https://github.com/user-attachments/assets/39ee2d1b-ad11-4dde-a7d1-babdbdac6fdb" />

<img width="1042" height="271" alt="image" src="https://github.com/user-attachments/assets/4a5468c9-65c4-42bc-ae4c-1cfac92a0a3e" />


###  Sobel Edge Detector
- Detects edges in horizontal and vertical directions  
- Produces gradient-based edge map  

###  Prewitt Edge Detector
- Similar to Sobel but simpler kernel  
- Detects directional edges  

###  Roberts Edge Detector
- Detects edges using diagonal gradients  
- Sensitive to noise  

###  Laplacian Edge Detector
- Detects edges using second-order derivatives  
- Highlights rapid intensity changes  

###  Canny Edge Detector
- Multi-stage edge detection  
- Produces clean and thin edges  

---

## Result

Thus, edges are successfully detected using Sobel, Prewitt, Roberts, Laplacian, and Canny edge detection techniques. Each method highlights edges differently based on gradient and intensity variations, improving feature extraction and analysis.
