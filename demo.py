import cv2
import numpy as np
import HandTrackingModule as htm
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

# Initializing the canvas on which we will draw upon
canvas = None

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800
totalFingerCount=1
detector=htm.handDetector(detectionCon=0.7)
tipIds = [4,8,12,16,20]
iterate = 1
while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    # Initilize the canvas as a black image of same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also its size is bigger than the noise threshold.
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
        # This is true when we writing for the first time or when writing again when the pen had disapeared from view.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            # Draw the line on the canvas
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
    # Merge the canvas and the frame.
    frame = cv2.add(frame,canvas)
    img = detector.findHands(frame)
    lmList = detector.findPosition(img, draw=False)
    flag = 0
    totalFingerCount =0

    if(len(lmList)!=0):
        fingers =[]
            

        for id in range(1,5):
            if(lmList[tipIds[id]][2] < lmList[tipIds[2]-2][2]):
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingerCount = fingers.count(1)

        

    
    
    
    # Optionally stack both frames and show it.
    #stacked = np.hstack((canvas,frame))
    cv2.putText(canvas, f'ENJOY LEARNING IN A NEW WAY', (10,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    if(totalFingerCount>0 and iterate == 1):
        writtenText = totalFingerCount
        flag = 1
        iterate = 0
    
    if(flag ==1):
        cv2.rectangle(canvas,(20,225), (170,425), (0,255,0), cv2.FILLED)
        cv2.putText(canvas,str(writtenText),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    
    
    k = cv2.waitKey(1) & 0xFF
    if k != ord('p'):
        cv2.imshow('Trackbars',cv2.resize(canvas,None,fx=0.6,fy=0.6))
    else:
        cv2.imshow('Image',img)

    if k == 27:
        break
    if k == ord('s'):
        cv2.imwrite('frame.jpg',canvas)
        
    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None
        iterate = 1
    if k == ord('w'):
        cv2.putText(canvas,"Correct",(100,150),cv2.FONT_HERSHEY_PLAIN,10,(0,255,0),20)
    elif k == ord('e'):
        cv2.putText(canvas,"Wrong",(100,150),cv2.FONT_HERSHEY_PLAIN,10,(0,0,255),20)

    

    


cv2.destroyAllWindows()
cap.release()
