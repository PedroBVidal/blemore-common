import os, sys
import numpy as np
from tqdm import tqdm
import argparse
import re
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-size', type=int, default=16)
    args = parser.parse_args()
    return args


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def get_imediate_subdirs_paths(source_dir):
    subdirs = [
        os.path.join(source_dir, d) 
        for d in os.listdir(source_dir) 
        if os.path.isdir(os.path.join(source_dir, d))
    ]
    subdirs.sort(key=natural_sort_key)
    return subdirs


def get_files_names(subdir_path, suffix=".png", sequence_size=16):   # sequence_size <= 0 means ALL FILES
    files_tmp = [
        f for f in os.listdir(subdir_path) 
        if os.path.isfile(os.path.join(subdir_path, f)) and f.endswith(suffix)
    ]
    files_tmp.sort(key=natural_sort_key)
    if sequence_size <= 0:
        return files_tmp
    assert len(files_tmp) >= sequence_size, f"Error, len(files_tmp) ({len(files_tmp)}) < sequence_size ({sequence_size})"

    # return all files, with on subsampling
    if len(files_tmp) == sequence_size:
        return files_tmp

    indices_to_select = np.linspace(0, len(files_tmp)-1, sequence_size, dtype=int)
    files = [files_tmp[i] for i in indices_to_select]
    assert len(files) == sequence_size, f"Error, len(files) ({len(files)}) == sequence_size ({sequence_size})"
    return files


def load_normalize_transpose_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # from (112,112,3) to (3,112,112)
    img = ((img / 255.) - 0.5) / 0.5    # normalize data to range [-1, 1]
    return img.astype(np.float32)


def aggregate_sequence_imgs_and_save_npz(source_dir, output_path, suffix=".png", sequence_size=16):
    all_subdirs_videos_paths = get_imediate_subdirs_paths(source_dir)
    # print('all_subdirs_videos_paths:', all_subdirs_videos_paths)
    files_first_dir = get_files_names(all_subdirs_videos_paths[0], suffix, sequence_size)
    img_first_frame = cv2.imread(os.path.join(all_subdirs_videos_paths[0], files_first_dir[0]))
    all_frames_array_shape = (len(all_subdirs_videos_paths), sequence_size, img_first_frame.shape[2], img_first_frame.shape[0], img_first_frame.shape[1])
    
    print(f"Preallocating array for all frames:", all_frames_array_shape)
    X = np.zeros(all_frames_array_shape, dtype=np.float32)
    all_filenames = [None] * len(all_subdirs_videos_paths)

    for idx_subdir_video, subdir_video_path in enumerate(all_subdirs_videos_paths):
        frames_filenames = get_files_names(subdir_video_path, suffix, sequence_size)
        video_name = os.path.basename(subdir_video_path)
        for idx_fname, fname in enumerate(frames_filenames):
            print(f"video {idx_subdir_video}/{len(all_subdirs_videos_paths)} ({idx_subdir_video/len(all_subdirs_videos_paths)*100:.1f}%) '{video_name}'  - ",
                  f"frame {idx_fname}/{len(frames_filenames)}  - ",
                  f"sequence_size {sequence_size}", end='\r')

            if not fname.endswith(suffix):
                continue

            path = os.path.join(subdir_video_path, fname)
            try:
                img_transp = load_normalize_transpose_img(path)
                X[idx_subdir_video,idx_fname:] = img_transp

            except Exception as e:
                print(f"Failed: {fname} — {e}")

        all_filenames[idx_subdir_video] = video_name
        print()

    filenames = np.array(all_filenames)
    
    print(f"Saving all stacked data X {X.shape}: '{output_path}'")
    # np.savez(output_path, X=X, filenames=filenames)
    np.savez_compressed(output_path, X=X, filenames=filenames)
    print(f"Saved: {output_path} (X shape: {X.shape}, {len(filenames)} filenames)")



def aggregate_sequence_features_and_save_npz(source_dir, output_path, suffix=".npy", sequence_size=16):
    all_features = []
    all_filenames = []

    for fname in tqdm(os.listdir(source_dir)):
        if not fname.endswith(suffix):
            continue

        path = os.path.join(source_dir, fname)
        x = np.load(path)
        x = np.squeeze(x)
        if x.ndim != 2:
            print(f"Skipping {fname}: shape {x.shape}")
            continue

        assert len(x) >= sequence_size, f"Error, len(x) ({len(x)}) < sequence_size ({sequence_size})"
        indices_to_select = np.linspace(0, len(x)-1, sequence_size, dtype=int)
        assert len(indices_to_select) == sequence_size, f"Error, len(indices_to_select) ({len(indices_to_select)}) != sequence_size ({sequence_size})"
        agg = x[indices_to_select,:]
        
        # agg = np.concatenate([
        #     x.mean(axis=0),
        #     x.std(axis=0),
        #     np.percentile(x, 10, axis=0),
        #     np.percentile(x, 25, axis=0),
        #     np.percentile(x, 50, axis=0),  # median
        #     np.percentile(x, 75, axis=0),
        #     np.percentile(x, 90, axis=0),
        # ])

        all_features.append(agg)
        all_filenames.append(fname.replace(suffix, ""))

    X = np.stack(all_features)
    filenames = np.array(all_filenames)

    np.savez(output_path, X=X, filenames=filenames)
    print(f"Saved: {output_path} (X shape: {X.shape}, {len(filenames)} filenames)")



def main():
    args = parse_args()

    # base_static_dir = "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data"
    base_static_dir = "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data"

    # os.makedirs(base_static_dir, exist_ok=True)

    encoding_paths = {
        # "openface": "/home/tim/Work/quantum/data/blemore/encoded_videos/openface_npy/",
        # "imagebind": "/home/tim/Work/quantum/data/blemore/encoded_videos/ImageBind/",
        # "clip": "/home/tim/Work/quantum/data/blemore/encoded_videos/CLIP_npy/",
        # "dinov2": "/home/tim/Work/quantum/data/blemore/encoded_videos/dynamic_data/DINOv2_first_component/",
        # "videoswintransformer": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoSwinTransformer/",
        # "videomae": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoMAEv2_reshaped/",
        # "hubert": "/media/user/Seagate Hub/mixed_emotion_challenge/audio_encodings/hubert_large/",
        # "wavlm": "/media/user/Seagate Hub/mixed_emotion_challenge/audio_encodings/wavlm_large/",
        # "hicmae": "/home/tim/Work/quantum/data/blemore/encoded_videos/original_encodings/HiCMAE"

        # "imagebind": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/ImageBind_train/",
        "seq_imagebind": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/ImageBind_train/",
        # "videomae":  "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/VideoMAEv2_train/",

        # "rawimgs112x112": "/home/pbqv20/BlEmoRe_backup/data_frames_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs_112x112/train/all_parts"
        # "seq_rawimgs112x112": "/home/pbqv20/BlEmoRe_backup/data_frames_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs_112x112/train/all_parts"
    }

    # suffix=".png"    # for "rawimgs112x112", "seq_rawimgs112x112"
    suffix=".npy"      # for "imagebind" and other pre extracted features

    for encoder, path in encoding_paths.items():
        output_path = os.path.join(base_static_dir, f"{encoder}_SEQUENCE_features_sequence={args.sequence_size}.npz")
        print(f"Processing {encoder} from {path}...")    
        if suffix == ".png" or suffix == ".jpg" or suffix == ".jpeg":
            aggregate_sequence_imgs_and_save_npz(path, output_path, suffix=suffix, sequence_size=args.sequence_size)
        elif suffix == ".npy":
            aggregate_sequence_features_and_save_npz(path, output_path, suffix=suffix, sequence_size=args.sequence_size)
        print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    main()