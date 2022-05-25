from cv2 import VideoWriter, VideoWriter_fourcc, imwrite
import os

def create_video(frames, video_file, img_dir=None):
    width, height, _ = frames[0].shape
    fps = 60
    # fourcc = VideoWriter_fourcc(*'MP42')
    fourcc = VideoWriter_fourcc(*'MP4V')
    video = VideoWriter(video_file, fourcc, float(fps), (width, height))

    for i in range(len(frames)):
        if img_dir is not None:
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            img_path = os.path.join(img_dir, f"step_{i}.jpg")
            imwrite(img_path, frames[i])
        video.write(frames[i])
    video.release()
