#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 얼굴 감지 초기화 (손에서 얼굴로 변경)
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
)

# =========================================
# 📸 동영상 스트림 연결 (웹캠 또는 로컬 동영상 파일)
# =========================================
# 웹캠을 쓰시려면 0, 로컬 파일(face.mp4 등)을 쓰시려면 파일명을 넣어주세요
# 'face.mp4' 파일로 연결
cap = cv2.VideoCapture("face.mp4")

if not cap.isOpened():
    print("❌ 영상을 열 수 없습니다.")
    exit()

print("📷 얼굴 감지 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 스트림을 종료합니다.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 👤 얼굴 감지 결과 표시 (손 랜드마크 대신 얼굴 감지 박스 표시)
    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

    # 화면 표시
    cv2.imshow('🤖 Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()