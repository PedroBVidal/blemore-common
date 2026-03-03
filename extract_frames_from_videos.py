import cv2
import argparse
import os
from pathlib import Path

def extract_frames(input_path, output_path, video_exts, image_ext):
    input_base = Path(input_path)
    output_base = Path(output_path)

    for video_file in input_base.rglob('*'):
        if video_file.suffix.lower() in video_exts:
            
            relative_path = video_file.relative_to(input_base).with_suffix('')
            clean_rel_path = Path(*[part.replace(" ", "_") for part in relative_path.parts])
            
            target_folder = output_base / clean_rel_path
            target_folder.mkdir(parents=True, exist_ok=True)

            print(f"Processing: {video_file.name} -> {target_folder}")

            cap = cv2.VideoCapture(str(video_file))
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_filename = f"{os.path.basename(target_folder)}_frame_{frame_count:06d}{image_ext}"
                frame_path = target_folder / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1

            cap.release()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-path", type=str, required=True, default='/home/pbqv20/BlEmoRe_backup/data/train/all_parts', help="Folder containing subfolders and/or videos")
    parser.add_argument("--input-ext", nargs='+', default=['.mov'], help="List of valid video extensions (e.g., .mp4 .mov)")
    parser.add_argument("--output-path", type=str, required=True, default='/home/pbqv20/BlEmoRe_backup/data_frames/train/all_parts', help="Folder to save extracted frames")
    parser.add_argument("--output-ext", type=str, default='.jpg', help="Image output extension (default: .jpg)")
    args = parser.parse_args()

    video_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in args.input_ext]
    print('video_exts:', video_exts)
    image_ext = args.output_ext if args.output_ext.startswith('.') else f'.{args.output_ext}'

    extract_frames(args.input_path, args.output_path, video_exts, image_ext)
    print("\nExtraction complete!")

if __name__ == "__main__":
    main()