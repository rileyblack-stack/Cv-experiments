import cv2
import numpy as np

# Load reference image and scene image
reference = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)
scene = cv2.imread("scene.jpg", cv2.IMREAD_GRAYSCALE)

if reference is None or scene is None:
    print("Error: Could not load one or both images.")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(reference, None)
kp2, des2 = sift.detectAndCompute(scene, None)

if des1 is None or des2 is None:
    print("Error: Could not compute descriptors.")
    exit()

# FLANN matcher parameters
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

MIN_MATCH_COUNT = 10

if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if matrix is not None:
        h, w = reference.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        scene_color = cv2.imread("scene.jpg")
        result = cv2.polylines(scene_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Draw matches
        matched_vis = cv2.drawMatches(
            reference, kp1,
            scene, kp2,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow("Detected Object", result)
        cv2.imshow("Feature Matches", matched_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Homography could not be computed.")
else:
    print(f"Not enough matches found. Found {len(good_matches)} matches.")
