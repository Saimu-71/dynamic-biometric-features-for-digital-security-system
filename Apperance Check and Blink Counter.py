

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# cap = cv2.VideoCapture('Video.mp4')
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

# idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243,463,286,257,260,359,339,253,256]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)
eye_state="open"
while True:

    # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5,color, cv2.FILLED)
        up = face[10]
        down = face[152]

        dis, _= detector.findDistance(up, down)
        if dis<165 :
            cvzone.putTextRect(img, f"dis = {dis}   Come Closer or Be appeared Physically", (50, 100),1,1,
                               colorR=color)
            imgStack = cvzone.stackImages([img, img], 2, 1)
            # cv2.imshow("Image", img)
            # continue
        # leftUp = face[159]
        # leftDown = face[23]
        # leftLeft = face[130]
        # leftRight = face[243]
        else:
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            rightUp = face[386]
            rightDown = face[253]
            rightLeft = face[463]
            rightRight = face[359]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
            lenghtVer1, _ = detector.findDistance(rightUp, rightDown)
            lenghtHor1, _ = detector.findDistance(rightLeft, rightRight)



            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
            cv2.line(img, rightUp, rightDown, (0, 200, 0), 3)
            cv2.line(img, rightLeft, rightRight, (0, 200, 0), 3)
            ratio = int((lenghtVer / lenghtHor) * 100)
            ratio1 = int((lenghtVer1 / lenghtHor1) * 100)
            ma=min(ratio1,ratio)
            ratioList.append(ma)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)

            # if ratioAvg < 36.5 and counter == 0 and eye_state=="open":
            if ratioAvg < 36.5 and counter == 0 and eye_state == "open":
                blinkCounter += 1
                color = (0,200,0)
                counter = 1
                eye_state="closed"
            elif ratioAvg>36.5 or ratioAvg==36.5:
                eye_state="open"
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255,0, 255)


            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 50),
                               colorR=color)
            cvzone.putTextRect(img, f"dis:{dis}", (50, 100), 1, 1,
                               colorR=color)

            imgPlot = plotY.update(ratioAvg, color)
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    key = cv2.waitKey(5)

    if key == ord('x') or key == ord('X'):
        break

cap.release()
cv2.destroyAllWindows()

