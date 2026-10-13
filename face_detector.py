#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 얼굴 감지(Face Detection) 초기화
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 검출기 설정 (min_detection_confidence: 탐지 신뢰도)
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
)

# =========================================
# 📸 비디오(또는 카메라) 연결
# =========================================
# cap = cv2.VideoCapture(0)            # 기본 카메라 사용 시
cap = cv2.VideoCapture(r"C:\Users\HKNU\Desktop\MediaPipe\face.mp4")   # 동영상 파일 사용 시

print("📷 얼굴 감지 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 파일이나 카메라 연결을 확인하세요.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 👤 얼굴 랜드마크 및 바운딩 박스 표시
    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

    # 화면 표시
    cv2.imshow('🤖 MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()