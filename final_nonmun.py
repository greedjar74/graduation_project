import cv2
import numpy as np
import time
import pygame

# Yolo 로드
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
classes = []

# 학습 파일 open
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(*classes) # 클래스들 출력

warning = cv2.imread("warning.png", cv2.IMREAD_UNCHANGED)
warning_resize = cv2.resize(warning, dsize=(1000, 1000))

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# opencv python 코딩 기본 틀
# 카메라 영상을 받아올 객체 선언 및 설정(영상 소스, 해상도 설정)
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prevTime = 0

warning_time = 2
detect_time = time.time()
war = 0

# 무한루프
while True:
    ret, frame = capture.read() # 카메라로부터 현재 영상을 받아 frame에 저장, 잘 받았다면 ret가 참
    # frame(카메라 영상)을 original 이라는 창에 띄워줌 
    # 이미지 가져오기
    img = cv2.resize(frame, None, fx=1, fy=1)
    height, width, channels = img.shape
    
    curTime = time.time() # fps체크
    sec = curTime - prevTime # 현재 시각 - 처음 시각
    prevTime = curTime
    fps = 1/(sec) # fps
    fps_str = "FPS: %.2f"%fps
    print(fps_str) # fps출력

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []

    eye_cnt = 0 # 눈의 개수 체크 변수

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] # 정확도
            if confidence > 0.5:
                eye_cnt += 1 # 눈의 개수 1증가
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if eye_cnt >= 1 : # 눈이 1개 이상인 경우 -> 전방주시 정상으로 간주
        detect_time = time.time() # 현재 시간 저장
        
    # 현재 시각에서 마지막으로 전방주시가 정상이었던 시각을 뺀 값이 일정 시간을 넘어간 경우 전방주시 불량으로 판단
    if time.time() - detect_time > warning_time: #1차 경고 출력
        print("경고 (종료: 2초)") # 경고 2초간 출력
        music_file = "99E6EB435CDE913E23.mp3" # 경고음 경로 생성
        pygame.mixer.init() # mixer생성
        pygame.mixer.music.load(music_file) # 경고음 파일 로드
        pygame.mixer.music.play(-1) # 경고음 재생 -> -1으로 설정하여 무한 반복
        clock = pygame.time.Clock()
        clock.tick()
        war = 1 # 경고가 있었다는 정보를 저장
        # Display the second image for 2 seconds
        cv2.imshow("WARNING", warning_resize) # 경고 이미지 출력
        cv2.waitKey(2000)#2초간 경고창 출력(경고중에는 눈의 인식여부 상관 없음 일단 한번 걸렸기 때문에 경고해줘야함)
        cv2.destroyWindow("WARNING") # 2초 후 창 닫기 -> 사용자의 피드백 필요 없음
        pygame.mixer.quit() # mixer삭제
        
        detect_time2 = time.time() # 전방주시 불량 알림 후 1초간 전방주시가 정상인지 판단에 사용
        detect_time = time.time() # detect_time초기화
  
    # 이전에 경고가 있었고 마지막으로 전방주시가 정상이었던 시각과 현재 시각의 차이가 1초를 넘어간 경우
    elif war == 1 and time.time() - detect_time > 1: # 2차 경고 출력
        print("경고 (종료: 사용자 피드백)")
        
        music_file = "99E6EB435CDE913E23.mp3" # 경고음 경로 생성
        pygame.mixer.init() # mixer 생성
        pygame.mixer.music.load(music_file) # 경고음 파일 로드
        pygame.mixer.music.play(-1) # 경고음 재생 -> 무한반복
        clock = pygame.time.Clock()
        clock.tick()

        cv2.imshow("WARNING", warning_resize) # 경고 이미지 출력
        key = cv2.waitKey(0)
        
        # 사용자의 피드백이 있는 경우 경고 종료
        if key == ord('t'):  # 키보드의 t 를 누르면 무한루프가 멈춤
            cv2.destroyWindow("WARNING")
            pygame.mixer.music.stop()
        war = 0
        #앞선 경고에도 불구하고 다시 한번 더 걸렸기 때문에 운전자가 직접 특정 버튼을
        #눌러야 경고창이 사라지도록 함, 특정한 경고가 울리는 시간을 정하지 않고
        #운전자가 버튼을 누르면 경고가 꺼지도록 알고리즘을 정함
        
    # 이전에 경고가 있은 후 전방주시가 1초간 정상인 경우 -> 초기화
    if war == 1 and detect_time - detect_time2 > 1:
        war = 0
        detect_time2 = 0
        
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
            
    cv2.putText(img, fps_str, (0, 25), font, 1, (0, 255, 0))
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == ord('q'):  # 키보드의 q 를 누르면 무한루프가 멈춤
        break

capture.release()                   # 캡처 객체를 없애줌
cv2.destroyAllWindows()             # 모든 영상 창을 닫아줌