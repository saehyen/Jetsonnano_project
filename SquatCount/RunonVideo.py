import cv2
import ffmpeg
import numpy as np
import time

from tensorflow.keras.models import load_model

# 모델 경로 설정
modelPath = 'C:/Users/HUSTAR12/Desktop/Squat/saved/saved11.h5'


# 비디오 회전 확인(불필요)
def checkVideoRotation(videoPath):
    metadata = ffmpeg.probe(videoPath)

    code = None
    
    #if int(metadata['streams'][0]['tags']['rotate']) == 90:
    #    code = cv2.ROTATE_90_CLOCKWISE
    #elif int(metadata['streams'][0]['tags']['rotate']) == 180:
    #    code = cv2.ROTATE_180
    #elif int(metadata['streams'][0]['tags']['rotate']) == 270:
    #    code = cv2.ROTATE_90_COUNTERCLOCKWISE

    return code
# 비디오 편집 설정
def frameForModel(frame):
    f = cv2.resize(frame, (128, 128))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
    f = np.expand_dims(f, axis = 0)
    f = np.expand_dims(f, axis = -1)
    f /= 255.
    
    return f
    
# cap = cv2.VideoCapture(videoPath)
# 모델 불러오기
model = load_model(modelPath)   

# 폰트, 선 설정
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 2
fontColor              = (120, 135, 0)
lineType               = 10

# 스쿼트 상태 초기화
count = 0
currentState = [-1, 0]
minStrength = 3
canIncreaseCount = False

# cap : 프레임별 캡쳐
cap = cv2.VideoCapture(0)
                # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('Frame', 480, 640)   
# 회전코드
#rotateCode = checkVideoRotation(videoPath)

# 비디오 저장 경로 및 세부설정 조절
out = cv2.VideoWriter('C:/Users/HUSTAR12/Desktop/Squat/output2.mp4', -1, 30.0, (1080, 1920))

# 비디오를 찾지 못하면 에러
if not cap.isOpened():
    print("Error while opening the video:")
start_time = time.time()
timeset=True
timeset2=True
Framenum = 0
category_ = 'None'
state_ = -1
# 프레임이 존재할때
while cap.isOpened(): 
    ret, frame = cap.read()
    Framenum+=1
    if ret:
 #       if rotateCode is not None:
 #          frame = cv2.rotate(frame, rotateCode)
         
        pred = model(frameForModel(frame)).numpy()
       #5초간 멈췄다 시작하기
        
        c = np.argmax(pred)
        # 초기세팅
        category = 'None'
        state = -1
        # 정확도가 50퍼센트 이상일때
        if pred[0][c] >= 0.5 and Framenum % 5  == 0 :
            if c == 0 :
                category = "Sit_"
                state = 0
            elif c == 1:
                category = "Sit"
                state = 1
            elif c == 2:
                category = "Stand"
                state = 2
        else :
            category = category_
            state = state_
        if currentState[0] == state:
            currentState[1] += 1
        else:
            currentState = [state, 1]
            
        if currentState[1] > minStrength:
            if currentState[0] == 1:
                canIncreaseCount = True
            elif currentState[0] == 2 and canIncreaseCount:
                count += 1
                canIncreaseCount = False
        # 화면크기 실험중  
        # 젯슨나노 2560x1440 ,1920*1080
        frame = cv2.resize(frame, (2560,1400))  
        
        category_ = category
        state_ = state
        # 스쿼트 개수, 상태 표시
        if time.time()-start_time >= 4 :
            cv2.putText(frame,
                        category, 
                        (20, 50), 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
            
            cv2.putText(frame,
                        "Time : " +str((int)(time.time()-start_time)-4) + "sec", 
                        (20, 150), 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(frame,
                        "Count: " + str(count), 
                        (20, 250), 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)

            
        if time.time()-start_time <= 4 :
             cv2.putText(frame,
                    str((int)(5-(time.time()-start_time))), 
                    (1200, 800), 
                    font, 
                    5,
                    (255,0,0),
                    lineType)
             
             cv2.putText(frame,
                    "Ready", 
                    (1000, 600), 
                    font, 
                    6,
                    (255,0,0),
                    lineType)
        
        cv2.imshow('Frame', frame)
        if timeset == True :
            start_time = time.time()
            timeset = False
            
        # 5초간 카운트 제한
        if time.time()-start_time >= 5 and timeset2 != False :
            #print("5초지남")
            count = 0
            timeset2 = False
        
        # 비디오 저장
        #out.write(frame)
        # 실시간 종료 : q
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 실행하는곳
cap.release()
out.release()
cv2.destroyAllWindows()