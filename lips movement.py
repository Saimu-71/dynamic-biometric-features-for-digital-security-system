import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_lips = mp.solutions.face_mesh

# cap = cv2.VideoCapture(r"C:\Users\HP\Pictures\Camera Roll\Lip movement.mp4")
cap = cv2.VideoCapture(0)
ratioList=[]
counter=0
# while cap.isOpened():
while cap.read():
    success, image = cap.read()

    if not success:
        # print("Ignoring empty camera frame.")
        continue

    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = mp_lips.FaceMesh().process(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    text = "Lips are open"
    text1= "Lips are closed"
    position = (50, 50)  # (x, y) coordinates of the top-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    color = (0, 255, 0)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            upper_lip_center = [face_landmarks.landmark[13].x, face_landmarks.landmark[13].y]
            lower_lip_center = [face_landmarks.landmark[14].x, face_landmarks.landmark[14].y]
            lip_distance = lower_lip_center[1] - upper_lip_center[1]
            # upper_lip_center1 = [face_landmarks.landmark[12].x, face_landmarks.landmark[12].y]
            # lower_lip_center1 = [face_landmarks.landmark[15].x, face_landmarks.landmark[15].y]
            # lip_distance1 = lower_lip_center[1] - upper_lip_center[1]
            ratioList.append((lip_distance))
            if len(ratioList) > 2:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)
            # if ratioAvg > 0.03 :
            if ratioAvg > 0.01:
                # cv2.putText(image, f"{text},dis={str(lip_distance1)}", position, font, scale, color, thickness=2)
                cv2.putText(image, f"{text}", position, font, scale, color, thickness=2)
            else:
                # cv2.putText(image, f"{text1},dis={str(lip_distance1)}", position, font, scale, color, thickness=2)
                cv2.putText(image, f"{text1}", position, font, scale, color, thickness=2)


            # mp_drawing.draw_landmarks(image, face_landmarks, mp_lips.FACEMESH_CONTOURS)

    cv2.imshow('MediaPipe Lips', image)
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break
    key=cv2.waitKey(25)
    if key == ord('x') or key == ord('X'):
        break

cap.release()
cv2.destroyAllWindows()
