import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)  # gave us the name of the files on that folder
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
cTime = 0

detector = htm.handDetector(detectionCon=0.85)
drawColor = (255, 0, 255)
brushThicknes = 15
xp, yp = 0, 0
# here is where we are going to draw
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # -  1 finger up -> draw
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:  # we are in the header
                if 250 < x1 < 450:
                    header = overlayList[3]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[0]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1+25), (x2, y2-25),
                          drawColor, cv2.FILLED)
        # -  2 fingers up -> selection mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), brushThicknes, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                brushThicknes = 50
            else:
                brushThicknes = 15

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThicknes)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThicknes)
            xp, yp = x1, y1

    # create a mask with the region colored
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    # add the mask and the image
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # knowing the mesures of our headers we can overlay the header to our image, because an image is just a matrix
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # print the fps
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255, 3))

    cv2.imshow("Image", img)
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()
