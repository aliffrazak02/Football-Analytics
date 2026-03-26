import cv2
from utils import read_video, save_video
from trackers import Tracker


def main():
    # Read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize tracker
    tracker = Tracker(model_path="models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # Save cropped image of a player
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        # crop bounding box from frame
        cropped_image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Save cropped image
        cv2.imwrite("output_videos/cropped_image.jpg", cropped_image)
        break

    # Draw annotations on video frames
    # Draw object tracks
    output_video_frames = tracker.draw_annotation(video_frames, tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.mp4")


if __name__ == "__main__":
    main()
