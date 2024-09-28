from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# path
occluded_image = "./test_images/with_occlusion/terrorists2.jpg"
unoccluded_image = "./test_images/without_occlusion/image1.jpg"

image = cv2.imread(occluded_image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
resp = RetinaFace.detect_faces(occluded_image)

# Draw rectangles around detected faces
for key in resp:
    face = resp[key]
    x1, y1, x2, y2 = face['facial_area']
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
