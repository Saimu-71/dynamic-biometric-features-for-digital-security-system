
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
countu=0
countd=0
countc=0
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT=[33]
L_H_RIGHT=[133]
R_H_LEFT=[362]
R_H_RIGHT=[263]
R_H_UP=[386]
R_H_DOWN=[374]
id=[33,133,362,263,386,374]

def euc_dis(po1,po2):
    x1,y1=po1.ravel()
    x2, y2 = po2.ravel()
    dis=math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dis
def iris_pos(center,right,left):
    cen2right_dis=euc_dis(center,right)
    total_dis=euc_dis(right,left)
    ratio=cen2right_dis/total_dis
    iris_pos=""
    if ratio<0.42:
        iris_pos="right"
    elif ratio>.42 and ratio <.57:
        iris_pos="center"
    else:
        iris_pos="left"
    return iris_pos,ratio

def iris_pos1(center,right,left):
    # cen2right_dis=euc_dis(center,right)
    # total_dis=euc_dis(right,left)
    cen2right_dis=abs(center-right)
    total_dis=abs(right-left)

    ratio=cen2right_dis/total_dis
    iris_pos=""
    #.42
    if ratio < 0.42:
        iris_pos = "up"
    elif ratio > .42 and ratio < .5:
        iris_pos = "middle"
    else:
        iris_pos = "down"
    return iris_pos,ratio

# cap = cv.VideoCapture(r"C:\Users\HP\Pictures\Camera Roll\Iris Tracking up.mp4")
# print(cap.get(cv.CAP_PROP_FRAME_COUNT))
# cap = cv.VideoCapture(r"D:\Camera Roll\lips  head iris\iris tracking down.mp4")
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # print(mesh_points.shape)
            # print(mesh_points)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            # print(center_left)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, 1, (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_right, 1, (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255,  255), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[R_H_UP][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[R_H_DOWN][0], 3, (0, 255, 255), -1, cv.LINE_AA)
            iris_position ,ratio = iris_pos(center_right,mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
            cv.putText(frame,
                       f"Iris pos: {iris_position} {ratio:.2f}",
                       (25,25),
                       cv.FONT_HERSHEY_PLAIN,
                       1.5,
                       (255,0,0),
                       1,
                       cv.LINE_AA,

                       )
            iris_position, ratio = iris_pos1(center_right[1], mesh_points[R_H_UP][0][1], mesh_points[R_H_DOWN][0][1])
            if iris_position=="down":
                countd+=1
            elif iris_position=="up":
                countu+=1
            else:
                countc+=1
            cv.putText(frame,
                       f"Iris pos: {iris_position} {ratio:.2f}",
                       (60,60),
                       cv.FONT_HERSHEY_PLAIN,
                       1.5,
                       (255, 0, 0),
                       1,
                       cv.LINE_AA,

                       )
        cv.imshow('img', frame)
        key = cv.waitKey(50)
        if key == ord('x') or key == ord('X'):
            break
    print("down=",countd)
    print("up=",countu)
    print("middle=",countc)
cap.release()
cv.destroyAllWindows()