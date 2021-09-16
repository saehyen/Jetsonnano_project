import cv2
import time
import numpy as np
import os

def proc(module, video):
    # 사용할 모델 지정
    if module == "coco":
        protoFile = "./model/coco.prototxt"
        weightsFile = "./model/coco.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],
                      [9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

        # <COCO module>
        #  0 - Nose
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
        # 14 - Right Eye
        # 15 - Left Eye
        # 16 - Right Ear
        # 17 - Left Ear
        
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
    
    # 동영상 데이터 로드, cv 규격에 맞는 데이터 크기로 변환
    inWidth = 368
    inHeight = 368
    threshold = 0.1
    input_source = "./input/" + video
    print(input_source)
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()
    height = 0
    
    #현재 상태를 저장
    Current_state_toggle = "Stand"
    #앉아있는 frame수 count
    sit_count = 0
    #서있는 frame수 count
    stand_count = 0
    #앉고, 서는 과정 routine을 시각적으로 보여주기 위해 해당 list에 저장.
    routine_list = []
    #init_setting 
    init_set = 0
    
    #비디오 저장
    vid_writer = cv2.VideoWriter('./output/' + video.split(".")[0] + "_"+ module + '.avi',
                                 cv2.VideoWriter_fourcc('M','J','P','G'),
                                 # 30이 기본배속 낮을수록 배속낮음
                                 30,
                                 (frame.shape[1],frame.shape[0]))
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    while cv2.waitKey(1) < 0:
        
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
        
        # toggle 조건 판별====================================================================
        # 서 있는 상태면 count 후 15 이상이면 toggle
        
        height = max(cal_distance(points[0], points[13]),height)
            
        if(Is_stand(height, points) and Current_state_toggle == "Sit"):
            stand_count += 1
            if(stand_count > 10):
                Current_state_toggle = "Stand"
                stand_count = 0
                #routine 리스트에 상태 저장
                routine_list.append("Stand")
                #현재까지 routine 리스트 출력
                print(routine_list)
                
        elif(Is_sit(height, points) and Current_state_toggle == "Stand"):
            sit_count += 1
            if(sit_count >10):
                Current_state_toggle = "Sit"
                sit_count = 0
                routine_list.append("Sit")
                print(routine_list)
        # toggle 조건 판별====================================================================
        
        
        # 점끼리 연결(그림 그리기)
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
        
        # 추가 : 현재 상태 체크 =================================================
        #현재 상태
        cv2.putText(frame, "Squat State = {} ".format(Current_state_toggle), (50, 50),
                   cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #Is_sit
        cv2.putText(frame, "Is_sit = {} ".format("true" if(Is_sit(height, points)) else "false"), (50, 100),
                   cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #Is_stand
        cv2.putText(frame, "Is_stand = {} ".format("true" if(Is_stand(height, points)) else "false"), (50, 150),
                   cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #squat routine
        cv2.putText(frame, "{} ".format(int(len(routine_list)/2)), (50, 200),
                   cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # 추가 : 현재 상태 체크 =================================================
        
        vid_writer.write(frame)
    vid_writer.release()

# 두 점사이 거리를 계산하는 함수
def cal_distance(points_1, points_2):
    if(points_1 != None and points_2 !=None):
        x_1 = points_1[0]
        x_2 = points_2[0]
        y_1 = points_1[1]
        y_2 = points_2[1]
        # abs = 절댓값
        diff_x = abs(x_1) - abs(x_2)
        diff_y = abs(y_1) - abs(y_2)
        
        return np.sqrt(diff_x**2 + diff_y**2)
    else:
        print("cal_distance has no points")
        return 0


#스쿼트 자세인지 확인
def Is_sit(height, points): #T: 앉은 자세, F: 서있을 때/어정쩡할때
    if (Angle(points)<105 and Angle(points)>65):
        if cal_distance(points[0], points[13])<height*0.75:
            return True
    else:
        return False

        
#일어선 자세인지 확인    
def Is_stand(height, points):
    if (Angle(points)>155 and Angle(points)<200):
        if cal_distance(points[0], points[13])>height*0.75:
            return True
    else:
        return False
    
def Angle(points):
    # 왼쪽 좌표 하나라도 None 이면 왼쪽 False
    pointA = points[11]
    pointB = points[12]
    pointC = points[13]
    pointD = points[8]
    pointE = points[9]
    pointF = points[10]
    left,right = True,True
    
    ymax = 1920
    # y좌표max값 = 800이라 가정
    if pointA != None :
        list(pointA)[1] += ymax
    if pointB != None :
       list(pointB)[1] += ymax
    if pointC != None :
        list(pointC)[1] += ymax
    if pointD != None :
        list(pointD)[1] += ymax
    if pointE != None :    
        list(pointE)[1] += ymax
    if pointF != None :
        list(pointF)[1] += ymax
       
    if pointA == None or pointB == None or pointC == None :
        left = False
    # 오른쪽 좌표하나라도 None 이면 오른쪽 False
    elif pointD == None or pointE == None or pointF == None :
        right = False
        
    # 둘다 안뜨면 False    
    if left == False and right == False :
        return False
    
    if left != False :
        # 왼쪽 좌표 계산
        x1, y1 = pointA
        x2, y2 = pointB
        x3, y3 = pointC
        v21 = (x1 - x2, y1 - y2)
        v23 = (x3 - x2, y3 - y2)
        dot = v21[0] * v23[0] + v21[1] * v23[1]
        det = v21[0] * v23[1] - v21[1] * v23[0]
        theta1 = np.rad2deg(np.arctan2(det, dot))
        if theta1 < 0 :
            theta1 = -theta1
        #print(theta1)
        
    if right != False :
        # 오른쪽 각도 계산
        x4, y4 = pointD
        x5, y5 = pointE
        x6, y6 = pointF
        v25 = (x4 - x5, y4 - y5)
        v27 = (x6 - x5, y6 - y5)
        dot = v25[0] * v27[0] + v25[1] * v27[1]
        det = v25[0] * v27[1] - v25[1] * v27[0]
        theta2 = np.rad2deg(np.arctan2(det, dot))
        
        if theta2 < 0 :
            theta2 = -theta2
        #print(theta2)
        
        if left != False and right != False :
            return int((theta1+theta2)/2)
        elif left != False and right == False :
            return int(theta1)
        else :
            return int(theta2)

if __name__ == '__main__':
    module = ["coco"]
    start_time = time.time()
    for i in module:
        for j in os.listdir("./input"):
            if j[-3:] in ['mp4','avi']:
                proc(i, j)
    print("{} ".format(time.time() - start_time))
