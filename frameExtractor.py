import cv2
import os

base_output_dir = './human_detection_dataset/train/'

videos = {
    'tall_rahul_person.MOV': '1',  # with_person
    'kayla_person.MOV': '1',
    'manvi_person.MOV': '1',
    'rajpreet_person.MOV': '1',
    'vindy_person.MOV': '1',
    'adnesh_person.mov': '1',
    'naveed_person.MOV': '1',
    'veda_person.mov': '1',
    'tall_rahul_apartment.MOV': '0',  # without_person
    'kayla_apartment.MOV': '0',
    'manvi_apartment.MOV': '0',
    'rajpreet_apartment.MOV': '0',
    'vindy_apartment.MOV': '0',
    'adnesh_apartment.MOV': '0',
    'naveed_apartment.MOV': '0',
    'veda_apartment.mov': '0'
}

max_frames = 500

for video_name, subdir in videos.items():
    print(f"Processing video {video_name}...")
    output_dir = os.path.join(base_output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True) 
    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_name}.")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    frame_count = 0
    saved_frames = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame or reached end of video.")
            break
        
        if frame_count % interval == 0 and saved_frames < max_frames:
            saved_frames += 1
            filename = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_{saved_frames}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame {saved_frames} at position {frame_count} for video {video_name}.")
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Finished extracting {saved_frames} frames from {video_name}.")

print("All video processing completed.")
