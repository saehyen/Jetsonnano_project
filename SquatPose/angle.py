
import numpy as np
# y max = 1920으로 가정
def angle(pointA,pointB,pointC,pointD,pointE,pointF):
    # 왼쪽 좌표 하나라도 None 이면 왼쪽 False
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
            return (theta1+theta2)/2
        elif left != False and right == False :
            return theta1
        else :
            return theta2
        
if __name__ == '__main__':
    #angle((50,500),(300,500),(300,800),None,None,None)
    #angle((600,200),(200,200),(200,600),None,None,None)
    #angle(None,None,None,(600,200),(200,200),(200,600))
    print(angle((50,500),(300,500),(300,800),(600,200),(200,200),(200,600)))