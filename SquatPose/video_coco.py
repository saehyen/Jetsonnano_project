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
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
        # keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho',
        #                     'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip',
        #                     'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
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
                                 30,
                                 (frame.shape[1],frame.shape[0]))
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []
        
        # 관절포인트마다 점 찍기
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
        
        # 점찍기 표시 및 점끼리 연결
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "Test Time = {:.2f} sec".format(time.time() - t), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        vid_writer.write(frame)
    vid_writer.release()



if __name__ == '__main__':
    module = ["coco"]
    for i in module:
        for j in os.listdir("./input"):
            if j[-3:] in ['mp4','avi']:
                proc(i, j)