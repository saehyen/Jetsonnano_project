import cv2
import time
import numpy as np
import sys
import os

def proc(module, img):
    # mpi모델 사용시 설정
    if module == "mpi":
        protoFile = "./model/mpi.prototxt"
        weightsFile = "./model/mpi.caffemodel"
        nPoints = 15
        # 부위 설정
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
    # cocomodel 사용시 적용
    else:
        protoFile = "./model/coco.prototxt"
        weightsFile = "./model/coco.caffemodel"
        nPoints = 18
        POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
    # 이미지 데이터 로드 및 하이퍼파라미터 설정
    print("./input/" + img)
    frame = cv2.imread("./input/" + img)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
    
    # 신경망을 cv 객체로 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    
    # cv 규격에 맞는 데이터(Blob)로 변환 및 데이터 입력
    t = time.time()
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)
    # 이미지 자세 추정 예측 수행
    output = net.forward()
    
    # 예측된 결과를 원본 이미지에 출력해서 점 찍기
    print("완료 : {:.3f}".format(time.time() - t))
    H = output.shape[2]
    W = output.shape[3]
    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)
    # 이미지에 생성된 Keypoint를 사전에 정한 규칙에 따라 연결
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        # 각 파트별 좌표 출력 (튜플)
        print(points[partA])
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            # 원 그리기
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # 스켈레톤 이미지 출력
    cv2.imwrite('./output/' + module + "_" + img, frame)

if __name__ == '__main__':
    module = ["mpi"]
    for i in module:
        for j in os.listdir("./input"):
        	if j[-3:] in ['jpg','png','peg']:
        		proc(i, j)