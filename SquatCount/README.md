# 실시간 스쿼트 Count


## 데이터 수집 방법
    - Upper (서있는 모습)
    - Middle (허벅지와 지면 각도 150도)
    - Lower ( 허벅지와 지면 각도 90도 )
    - 휴대폰 동영상 촬영, 3번째 Frame마다 각 폴더에 이미지 저장

## 모델 학습 ( 데이터셋_최종 ) 

    - 사용한 이미지 수 : 11978 (100%)
    - 학습에 사용 : 8982장 (75%)
    - 테스트에 사용 : 2996 (25%)
    - 이미지 크기 : 128 * 128
    - GrayScale
    
## 모델 학습 ( 모델 )
    - 사용한 프레임워크 : Keras
    - batch_size : 16
    - epochs : 10
    - 활성화함수 : layers.LeakyReLU(alpha = 0.1)
    - 손실함수 : categorical_crossentropy
    - 옵티마이저 : adam
    
## 모델 학습 ( 결과 ) 

    - 1차 : 테스트 데이터 : 10551(80:20)
<img src="README_Image\1차(8509_2042).png" width="700" height="370">

    - 2차 : 테스트 데이터 : 11978(80:20)
<img src="README_Image\2차(9606,2372).png" width="700" height="370">

    - 3차 : 테스트 데이터 : 11978(75:25)
<img src="README_Image\3차(8982,2996).png" width="700" height="370">

## 실제 작동 모습

    - 시작 전 작동 장면

<img src="README_Image\스크린샷1.png" width="700" height="370">

    - Upper

<img src="README_Image\스크린샷_Upper.png" width="700" height="370">

    - Middle

<img src="README_Image\스크린샷_Middle.png" width="700" height="370">


    - Lower(실패)

<img src="README_Image\스크린샷_Lower(실패).png" width="700" height="370">