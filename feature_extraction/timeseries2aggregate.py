import os
import sys
import numpy as np
from tqdm import tqdm
import re


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


def get_files_names(subdir_path, suffix=".png"):
    files_tmp = [
        f for f in os.listdir(subdir_path) 
        if os.path.isfile(os.path.join(subdir_path, f)) and f.endswith(suffix)
    ]
    files_tmp.sort(key=natural_sort_key)
    return files_tmp


def get_all_files_any_depth(directory_path, extension=".npy"):
    file_paths = []
    if not extension.startswith('.'):
        extension = f".{extension}"
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    file_paths.sort(key=natural_sort_key)
    return file_paths


def aggregate_bfm_and_save_npz(source_dir, output_path, suffix=".npy"):
    all_features = []
    all_filenames = []
    all_subdirs_videos_paths = get_imediate_subdirs_paths(source_dir)
    # files_first_dir = get_files_names(all_subdirs_videos_paths[0], suffix)
    # bfm_first_frame = np.read(os.path.join(all_subdirs_videos_paths[0], files_first_dir[0]))
    # all_frames_array_shape = (len(all_subdirs_videos_paths), sequence_size, img_first_frame.shape[2], img_first_frame.shape[0], img_first_frame.shape[1])

    for idx_subdir_video, path_subdir_video in enumerate(all_subdirs_videos_paths):
        video_name = os.path.basename(path_subdir_video)
        print(f"{idx_subdir_video}/{len(all_subdirs_videos_paths)} - {path_subdir_video}                      ", end='\r')
        # print("path_subdir_video:", path_subdir_video)
        files_paths = get_all_files_any_depth(path_subdir_video, extension=".npy")
        first_frame_feature = np.load(files_paths[0])
        video_frames_bfm = np.zeros((len(files_paths), first_frame_feature.shape[-1]), dtype=np.float32)
        for idx_file, path_file in enumerate(files_paths):
            x = np.load(path_file)
            video_frames_bfm[idx_file,:] = x

        agg = np.concatenate([
            video_frames_bfm.mean(axis=0),
            video_frames_bfm.std(axis=0),
            np.percentile(video_frames_bfm, 10, axis=0),
            np.percentile(video_frames_bfm, 25, axis=0),
            np.percentile(video_frames_bfm, 50, axis=0),  # median
            np.percentile(video_frames_bfm, 75, axis=0),
            np.percentile(video_frames_bfm, 90, axis=0),
        ])

        all_features.append(agg)
        all_filenames.append(video_name.replace(suffix, ""))

    print()

    X = np.stack(all_features)
    filenames = np.array(all_filenames)

    np.savez(output_path, X=X, filenames=filenames)
    print(f"Saved: {output_path} (X shape: {X.shape}, {len(filenames)} filenames)")



def aggregate_and_save_npz(source_dir, output_path, suffix=".npy"):
    all_features = []
    all_filenames = []

    for fname in tqdm(os.listdir(source_dir)):
        if not fname.endswith(suffix):
            continue

        path = os.path.join(source_dir, fname)
        try:
            x = np.load(path)
            x = np.squeeze(x)
            if x.ndim != 2:
                print(f"Skipping {fname}: shape {x.shape}")
                continue

            agg = np.concatenate([
                x.mean(axis=0),
                x.std(axis=0),
                np.percentile(x, 10, axis=0),
                np.percentile(x, 25, axis=0),
                np.percentile(x, 50, axis=0),  # median
                np.percentile(x, 75, axis=0),
                np.percentile(x, 90, axis=0),
            ])

            all_features.append(agg)
            all_filenames.append(fname.replace(suffix, ""))

        except Exception as e:
            print(f"Failed: {fname} — {e}")

    X = np.stack(all_features)
    filenames = np.array(all_filenames)

    np.savez(output_path, X=X, filenames=filenames)
    print(f"Saved: {output_path} (X shape: {X.shape}, {len(filenames)} filenames)")




def aggregate_fuse_and_save_npz(source_dir=[], output_path="", suffix=".npy"):
    all_features = []
    all_filenames = []

    for fname in tqdm(os.listdir(source_dir[0])):
        if not fname.endswith(suffix):
            continue

        path0 = os.path.join(source_dir[0], fname)
        path1 = os.path.join(source_dir[1], fname)
        # print("path0:", path0)
        # print("path1:", path1)
        try:
            x0 = np.load(path0)
            x1 = np.load(path1)

            x0 = np.squeeze(x0)
            x1 = np.squeeze(x1)
            
            if x0.ndim != 2:
                print(f"Skipping {fname}: shape {x0.shape}")
                continue

            agg0 = np.concatenate([
                x0.mean(axis=0),
                x0.std(axis=0),
                np.percentile(x0, 10, axis=0),
                np.percentile(x0, 25, axis=0),
                np.percentile(x0, 50, axis=0),  # median
                np.percentile(x0, 75, axis=0),
                np.percentile(x0, 90, axis=0),
            ])
            agg1 = np.concatenate([
                x1.mean(axis=0),
                x1.std(axis=0),
                np.percentile(x1, 10, axis=0),
                np.percentile(x1, 25, axis=0),
                np.percentile(x1, 50, axis=0),  # median
                np.percentile(x1, 75, axis=0),
                np.percentile(x1, 90, axis=0),
            ])
            # print("agg0.shape:", agg0.shape, "    agg1.shape:", agg1.shape)
            
            agg = np.concatenate([agg0, agg1])
            # print("agg.shape:", agg.shape)
            
            all_features.append(agg)
            all_filenames.append(fname.replace(suffix, ""))

        except Exception as e:
            print(f"Failed: {fname} — {e}")

    X = np.stack(all_features)
    filenames = np.array(all_filenames)

    np.savez(output_path, X=X, filenames=filenames)
    print(f"Saved: {output_path} (X shape: {X.shape}, {len(filenames)} filenames)")



def main():
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
        # "videomae":  "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/VideoMAEv2_train/",
        "bfm": "/home/pbqv20/BlEmoRe_backup/data_frames_HRN_3D_reconstruction/train/all_parts/",

        # "imagebind_wavlm": ["/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/ImageBind_train/",
        #                     "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/WavLM_large_train"]
        # "videomae_hubert": ["/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/VideoMAEv2_train/",
        #                     "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/HuBERT_large_train/"]
    }

    for encoder, path in encoding_paths.items():
        print(f"Processing {encoder} from {path}...")
        if type(path) is str:
            output_path = os.path.join(base_static_dir, f"{encoder}_static_features.npz")
            if encoder == "bfm":
                aggregate_bfm_and_save_npz(path, output_path, suffix=".npy")
            else:
                aggregate_and_save_npz(path, output_path, suffix=".npy")
        elif type(path) is list:
            output_path = os.path.join(base_static_dir, f"{encoder}_fused.npz")
            aggregate_fuse_and_save_npz(path, output_path, suffix=".npy")
        print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    main()