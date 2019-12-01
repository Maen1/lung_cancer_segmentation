import numpy as np
import cv2
cap = cv2.VideoCapture(0) # 0- Primary camera ,1- External camera

# colour to be masked - blue.
# param1 = [100,100,100] # Setting the lower pixel for blue (BGR)
# param2 = [150,150,255] # Setting the upper pixel for blue (BGR)

param1 = [30,30,30] # Setting the lower pixel for blue (BGR)
param2 = [150,150,150] # Setting the upper pixel for blue (BGR)

while(1):
    
    # Read the image
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # Do the processing
    lower = np.array(param1)    # Assigning the lower and upper index values (param1 and param2)
    upper = np.array(param2)
    mask  = cv2.inRange(hsv, lower, upper) # Masking of the image to produce a binary image.
    res   = cv2.bitwise_and(frame, frame, mask= mask) # The masked blue part of the image.
    # Show the image
    res = cv2.resize(res, (540, 540))
    mask = cv2.resize(mask, (540, 540))
    cv2.imshow('image',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    

    # End the video loop
    if cv2.waitKey(1) == 27:  # 27 - ASCII for escape key
        break

# Close and exit from camera
cap.release()
cv2.destroyAllWindows()