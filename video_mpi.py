import cv2
import time
import numpy as np
import os
import numpy as np

#enum 이나 #define

def proc(module, video):
    # 사용할 모델 지정
    if module == "coco":
        protoFile = "./model/coco.prototxt"
        weightsFile = "./model/coco.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],
                      [9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    elif module == "mpi":
        # 훈련된 network 셋팅
        protoFile = "./model/mpi.prototxt"
        weightsFile = "./model/mpi.caffemodel"
        nPoints = 15  #점의 개수
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], 
                      [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
        # <MPI module>
        #  0 - Head
        #  1 - Neck
        #  2 - Right Shoulder
        #  3 - Right Elbow
        #  4 - Right Wrist
        #  5 - Left Shoulder
        #  6 - Left Elbow
        #  7 - Left Wrist
        #  8 - Right Hip
        #  9 - Right Knee
        # 10 - Right Ankle
        # 11 - Left Hip
        # 12 - Left Knee
        # 13 - Left Ankle
        # 14 - Chest

    #frame 별 좌표 list 출력
    y_pos = []
    
    # 동영상 데이터 로드, cv 규격에 맞는 데이터 크기로 변환
    inWidth = 368
    inHeight = 368
    threshold = 0.1
    input_source = "./input/" + video
    print(input_source)
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()
    
    #비디오 저장
    vid_writer = cv2.VideoWriter('./output/' + video.split(".")[0] + "_"+ module + '.avi',
                                 cv2.VideoWriter_fourcc('M','J','P','G'),
                                 # 30이 기본배속 낮을수록 배속낮음
                                 10,
                                 (frame.shape[1],frame.shape[0]))
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break
        frameWidth = frame.shape[1]#input frame 의 너비(y 좌표)
        frameHeight = frame.shape[0]#input frame 의 높이(x 좌표)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []
        
        # 관절포인트마다 점 위치 계산
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            
                
            if prob > threshold : 
                # circle(그릴곳, 원의중심,반지름,색)
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)
        
        #좌표 list 출력
        y_pos.append(points[1][1])
        # 점끼리 연결
        #POSE_PAIR 은 연결된 점 두 쌍
        #points는 tuple 형식으로 좌표가 저장
        #ex) points = ((24,55),(42,123),(110,47),(45,12) . . . ,(30, 253))
        #    pair[]       0       1        2        3              14     
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                #화면에 선 표시
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                
                #화면에 점 표시
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "Test Time = {:.2f} sec".format(time.time() - t), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        vid_writer.write(frame)
        
    print(y_pos)
    
    vid_writer.release()
    #특정 point만 출력
    print_coordinate(y_pos)

# 두 점사이 거리를 계산하는 함수
def cal_distance(points_1, points_2):
    x_1 = points_1[0]
    x_2 = points_2[0]
    y_1 = points_1[1]
    y_2 = points_2[1]
    # abs = 절댓값
    diff_x = abs(x_1) - abs(x_2)
    diff_y = abs(y_1) - abs(y_2)
    
    return np.sqrt(diff_x**2 + diff_y**2)

#스쿼트 자세인지 확인
def Is_squat():
    pass


    # if(Is_squat()):
    #     squat_sit_count +=1 
    #     if(squat_sit_count ==15):
    #         squat_stand_count=0
    #         squat_sit_count = 0 
    #         sit_ +=1 
        
    # else(Is_stand()):
    #     squat_stand_count +=1
    #     if(squat_stand_count ==15):
    #         squat_stand_count=0
    #         squat_sit_count = 0 
            
    # if(sit_ ==15 and stand_ == 15)
    #     breaktime()
        
#일어선 자세인지 확인    
def Is_stand():
    pass

# y좌표 출력(이중 list)
def print_coordinate(y_pos):
    pass
    

if __name__ == '__main__':
    module = ["mpi"]
    for i in module:
        for j in os.listdir("./input"):
            if j[-3:] in ['mp4','avi']:
                proc(i, j)
