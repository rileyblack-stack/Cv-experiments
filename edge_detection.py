import cv2

# load image (make sure you have an image in your folder named image.jpg)
image = cv2.imread("image.jpg")

# check if image loaded
if image is None:
    print("Error: Image not found.")
    exit()

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# detect edges using Canny
edges = cv2.Canny(blurred, 50, 150)

# show images
cv2.imshow("Original", image)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
