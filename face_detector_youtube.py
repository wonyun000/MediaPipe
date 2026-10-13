#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp

# =========================================
# 🔗 YouTube 동영상 스트림 주소 추출 함수 (포맷 오류 방지 최적화)
# =========================================
def get_youtube_stream(url):
    ydl_opts = {
        'format': 'bv*+ba/b',  # 가장 확실한 조합형 포맷 또는 기본 포맷 선택
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # 만약 포맷이 여러 개 분리되어 있다면 첫 번째 비디오 URL 사용
        if 'url' in info:
            return info['url']
        elif 'formats' in info and len(info['formats']) > 0:
            for f in reversed(info['formats']):
                if f.get('url'):
                    return f['url']
        return info['entries'][0]['url']

# 📺 테스트할 유튜브 영상 링크 입력
youtube_url = 'https://www.youtube.com/shorts/jZjGsZmVVSI?feature=share'

print("⏳ 유튜브 영상 스트리밍 주소 추출 중...")
video_stream_url = get_youtube_stream(youtube_url)
print("✅ 스트리밍 주소 획득 완료!")

# =========================================
# 🧩 Mediapipe 얼굴 감지 초기화
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
)

# =========================================
# 📸 유튜브 스트림 연결
# =========================================
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ 유튜브 영상을 열 수 없습니다.")
    exit()

print("📷 유튜브 얼굴 감지 스트림 시작 — ESC를 눌러 종료합니다.")

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

    # 👤 얼굴 감지 결과 표시
    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

    # 화면 표시
    cv2.imshow('🤖 YouTube Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()