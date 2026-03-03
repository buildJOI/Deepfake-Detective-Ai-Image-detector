import cv2

def sample_frames(video_path: str, every_n: int = 30, max_frames: int = 10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames