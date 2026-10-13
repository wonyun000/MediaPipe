#!/usr/init/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp

# =========================================
# 🔗 YouTube 동영상 스트림 주소 추출 함수
# =========================================
def get_youtube_stream(url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best', # mp4 형식 중 가장 좋은 화질 선택
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# 📺 테스트할 유튜브 영상 링크 입력 (원하는 손 감지 유튜브 영상 주소로 변경 가능)
youtube_url = 'https://www.youtube.com/watch?v=JRRyYyiiuJQ'

print("⏳ 유튜브 영상 스트리밍 주소 추출 중...")
video_stream_url = get_youtube_stream(youtube_url)
print("✅ 스트리밍 주소 획득 완료!")

# =========================================
# 🧩 Mediapipe 초기화 (손 감지)
# =========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,       # 동영상 입력
    max_num_hands=2,              # 최대 손 개수
    min_detection_confidence=0.5, # 탐지 신뢰도
    min_tracking_confidence=0.5   # 추적 신뢰도
)

# =========================================
# 📸 유튜브 스트림 연결
# =========================================
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ 유튜브 영상을 열 수 없습니다.")
    exit()

print("📷 유튜브 손 감지 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 스트림을 종료합니다.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 손 검출 수행
    result = hands.process(image_rgb)

    # 🖐️ 손 랜드마크 표시
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # 화면 표시
    cv2.imshow('🖐️ YouTube Hand Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()