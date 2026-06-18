import os
import shutil
import subprocess
import cv2
from config import DATA_DIR, FRAMES_DIR, COOKIES_PATH


class Preprocessor:

    # Initializes the Preprocessor with video credentials, download paths, and target frame locations.
    def __init__(self, video_id: str, url: str):
        self.video_id   = video_id
        self.url        = url
        self.video_path = os.path.join(DATA_DIR, f"{video_id}.mp4")
        self.frames_dir = os.path.join(FRAMES_DIR, video_id)

    # Verifies if the raw MP4 video file is present in the download directory.
    def already_downloaded(self) -> bool:
        return os.path.exists(self.video_path)

    # Verifies if the frame extraction directory contains files from a previous run.
    def already_extracted(self) -> bool:
        return (
            os.path.exists(self.frames_dir)
            and len(os.listdir(self.frames_dir)) > 0
        )

    # Invokes yt-dlp to download the target YouTube video using specified quality constraints.
    def download(self) -> bool:
        if self.already_downloaded():
            print(f"[Preprocessor] Video already exists: {self.video_path}")
            return True

        os.makedirs(DATA_DIR, exist_ok=True)

        format_selector = (
            "bestvideo[height<=720][ext=mp4][vcodec^=avc]"
            "+bestaudio[ext=m4a]"
            "/best[ext=mp4]"
            "/bestvideo[height<=720]+bestaudio"
            "/best"
        )

        cmd = [ 
            "yt-dlp",
            "--cookies", COOKIES_PATH,
            "--js-runtimes", "node",
            "--sleep-interval", "8",
            "--max-sleep-interval", "20",
            "--retries", "10",
            "--fragment-retries", "10",
            "--concurrent-fragments", "1",
            "-f", format_selector,
            "--merge-output-format", "mp4",
            "-o", self.video_path,
            self.url,
        ]

        print(f"[Preprocessor] Downloading {self.url}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"[Preprocessor] yt-dlp failed for {self.url}:\n{result.stderr}"
            )

        print(f"[Preprocessor] Download complete: {self.video_path}")
        return True

    # Uses OpenCV to sample and save exactly one image frame per second of video duration.
    def extract_frames(self) -> list[str]:
        if self.already_extracted():
            print(f"[Preprocessor] Frames already exist for {self.video_id}, skipping extraction.")
            return sorted([
                os.path.join(self.frames_dir, f)
                for f in os.listdir(self.frames_dir)
                if f.endswith(".jpg")
            ])

        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"[Preprocessor] Could not open video: {self.video_path}"
            )

        fps   = cap.get(cv2.CAP_PROP_FPS)
        step  = round(fps)  
        count = 0
        saved = 0

        while True:
            success, frame = cap.read()
            if not success:
                break
            if count % step == 0:
                frame_id   = count // step
                frame_path = os.path.join(
                    self.frames_dir, f"frame_{frame_id:04d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                saved += 1
            count += 1

        cap.release()
        print(f"[Preprocessor] Extracted {saved} frames to {self.frames_dir}")

        return sorted([
            os.path.join(self.frames_dir, f)
            for f in os.listdir(self.frames_dir)
            if f.endswith(".jpg")
        ])

    # Coordinates the sequential execution of the video download and frame extraction pipeline.
    def run(self) -> list[str]:
        self.download()
        return self.extract_frames()