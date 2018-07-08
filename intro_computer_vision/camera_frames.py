"""
@author Wildo Monges
Practical introduction to camera manipulation using opencv
Book: Learning OpenCV 3 Computer Vision with Python Second Edition
"""
import cv2

clicked = False


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


camera_capture = cv2.VideoCapture(0)
cv2.namedWindow("MyWindow")
cv2.setMouseCallback("MiWindow", on_mouse)

print("Showing camera feed. Click window or press any key to stop")
success, frame = camera_capture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow("MyWindow", frame)
    success, frame = camera_capture.read()

cv2.destroyWindow("MyWindow")
camera_capture.release()