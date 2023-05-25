import cv2

print("OpenCV version:", cv2.__version__)
print("CUDA supported:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
