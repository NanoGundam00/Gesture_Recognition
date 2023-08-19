import cv2
import mediapipe as mp
import math

# 求解二维向量的角度
def vector_2d_angle(v1,v2):
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

# 获取对应手相关向量的二维角度,根据角度确定手势
def hand_angle(hand_):
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 二维约束的方法定义手势
def h_gesture(angle_list):
    thr_angle = 60.
    gesture_str = None
    if 65535. not in angle_list:
        if (angle_list[0]<thr_angle) and (angle_list[1]<thr_angle) and (angle_list[2]<thr_angle) and (angle_list[3]<thr_angle) and (angle_list[4]<thr_angle):
            gesture_str = "backwards"
        elif (angle_list[0]<thr_angle)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "turn around"
        elif (angle_list[0]>thr_angle)  and (angle_list[1]<thr_angle) and (angle_list[2]<thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "forward"
        elif (angle_list[0]>thr_angle)  and (angle_list[1]<thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]<thr_angle) and (angle_list[4]<thr_angle):
            gesture_str = "turn right"
        elif (angle_list[0]<thr_angle) and (angle_list[1]>thr_angle) and (angle_list[2]<thr_angle) and (angle_list[3]<thr_angle) and (angle_list[4]<thr_angle):
            gesture_str = "turn left"
    return gesture_str

def detect():
    
    # 初始化计时器
    start_time = cv2.getTickCount()
    frame_count = 0
    fps=0

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    cap = cv2.VideoCapture(0)  # 打开摄像头
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)  # 镜像翻转
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 计算帧率
        frame_count += 1
        end_time = cv2.getTickCount()
        elapsed_time = (end_time - start_time) / cv2.getTickFrequency()

        if elapsed_time > 1:  # 每秒更新一次帧率
            fps = frame_count / elapsed_time
            # 绘制帧率文本
            # 重置计时器和帧数
            start_time = end_time
            frame_count = 0

        
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # 画出关键点
                hand_local = []
                for i in range(21):  # 遍历21个关键点，记录每一个关键点的坐标(x,y)
                    x = hand_landmarks.landmark[i].x*frame.shape[1]
                    y = hand_landmarks.landmark[i].y*frame.shape[0]
                    hand_local.append((x,y))
                if hand_local:
                    angle_list = hand_angle(hand_local)
                    # for num,angle in enumerate(angle_list):
                    #     cv2.putText(frame,str(int(angle)),(0,50*(num+1)),0,1.3,(0,0,255),1)
                    gesture_str = h_gesture(angle_list)
                    cv2.putText(frame,gesture_str,(0,100),0,1.3,(0,0,255),3)
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    detect()
