import os
import sys
import numpy as np
from tqdm import tqdm


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

        # "imagebind_wavlm": ["/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/ImageBind_train/",
        #                     "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/WavLM_large_train"]
        "videomae_hubert": ["/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/VideoMAEv2_train/",
                            "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/HuBERT_large_train/"]
    }

    for encoder, path in encoding_paths.items():
        print(f"Processing {encoder} from {path}...")
        if type(path) is str:
            output_path = os.path.join(base_static_dir, f"{encoder}_static_features.npz")
            aggregate_and_save_npz(path, output_path, suffix=".npy")
        elif type(path) is list:
            output_path = os.path.join(base_static_dir, f"{encoder}_fused.npz")
            aggregate_fuse_and_save_npz(path, output_path, suffix=".npy")
        print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    main()