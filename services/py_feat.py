# import os
# import sys
# pyfeat_path = 'd:/coding_series/online-muesum/py-feat/feat'
# for root, dirs, files in os.walk(pyfeat_path):
#     print(f"Directory: {root}")
#     for file in files:
#         if file.endswith('.py'):
#             print(f"  - {file}")
#     # Only show first level for brevity
#     if root == pyfeat_path:
#         print(f"Subdirectories: {dirs}")
#         break
    
# sys.path.append('d:/coding_series/online-muesum/py-feat')

# from feat.detector import Detector

# # Initialize without arguments first to see if it works
# detector = Detector()

# # Check what arguments it accepts
# print(detector.__init__.__code__.co_varnames)
# # Or check the help documentation
# help(Detector)

import sys
import os

# Add the py-feat directory to Python's path
sys.path.append('d:/coding_series/online-muesum/py-feat')

# Import from feat module
from feat.detector import Detector

# Initialize detector
detector = Detector()

# Print available methods and attributes
print("\nDetector Methods and Attributes:")
for attr in dir(detector):
    if not attr.startswith('_'):
        print(f"- {attr}")

# Try to get documentation for key methods
print("\nDetector Method Documentation:")
methods_to_check = ["detect", "detect_faces", "detect_image"]
for method in methods_to_check:
    if hasattr(detector, method):
        print(f"\n{method}:")
        print(getattr(detector, method).__doc__)
    else:
        print(f"\n{method}: Not available")

print("\nTrying different methods with a sample image...")
# Create a small test image
import numpy as np
from PIL import Image
import cv2

# Create a simple test image (black square with white center)
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[25:75, 25:75] = 255  # White square in the middle

# Try different methods
try:
    if hasattr(detector, "detect"):
        print("\nTrying detector.detect()...")
        result = detector.detect(img)
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)[:10]}...")
except Exception as e:
    print(f"Error with detect(): {str(e)}")

try:
    if hasattr(detector, "detect_faces"):
        print("\nTrying detector.detect_faces()...")
        result = detector.detect_faces(img)
        print(f"Result type: {type(result)}")
        print(f"Result keys: {result.keys()}")
except Exception as e:
    print(f"Error with detect_faces(): {str(e)}")

# For a better test, save an actual face image
try:
    face_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    cv2.circle(face_img, (112, 100), 50, (255, 200, 200), -1)  # Simple face shape
    cv2.circle(face_img, (90, 90), 5, (0, 0, 0), -1)  # Left eye
    cv2.circle(face_img, (134, 90), 5, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(face_img, (112, 130), (30, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    cv2.imwrite("test_face.jpg", face_img)
    print("\nSaved test_face.jpg for testing")
    
    # Try with a real face image
    test_img = cv2.imread("test_face.jpg")
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # Try methods on this image
    for method in ["detect", "detect_faces"]:
        if hasattr(detector, method):
            try:
                print(f"\nTrying detector.{method}() with test face...")
                result = getattr(detector, method)(test_img_rgb)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error with {method}(): {str(e)}")
except Exception as e:
    print(f"Error creating test image: {str(e)}")