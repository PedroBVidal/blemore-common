import os
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
            # --- NEW LOGIC TO HANDLE (T, 1, D) SHAPES ---
            if x.ndim == 3 and x.shape[1] == 1:
                x = x.squeeze(axis=1)  # Converts (19, 1, 1408) to (19, 1408)

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


import os

def main():
    base_static_dir = "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data/"
    # base_static_dir = "/home/user/Work/quantum/data/blemore/encoded_videos/static_data"

    # os.makedirs(base_static_dir, exist_ok=True)

    encoding_paths = {
        # "openface": "/home/tim/Work/quantum/data/blemore/encoded_videos/openface_npy/",
        #"imagebind": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data/ImageBind_test",
        # "clip": "/home/tim/Work/quantum/data/blemore/encoded_videos/CLIP_npy/",
        # "dinov2": "/home/tim/Work/quantum/data/blemore/encoded_videos/dynamic_data/DINOv2_first_component/",
        # "videoswintransformer": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoSwinTransformer/",
        # "videomae": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoMAEv2_reshaped/",
        "videomaev2": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data/VideoMAEv2_test",
        "hubert": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data/HuBERT_large_test",
        "imagebind": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data/ImageBind_test",
        "wavlm": "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data/WavLM_large_test"
        # "wavlm": "/media/user/Seagate Hub/mixed_emotion_challenge/audio_encodings/wavlm_large/",
        #"hicmae": "/home/tim/Work/quantum/data/blemore/encoded_videos/original_encodings/HiCMAE"
    }

    for encoder, path in encoding_paths.items():
        output_path = os.path.join(base_static_dir, f"{encoder}_static_features.npz")
        print(f"Processing {encoder} from {path}...")
        aggregate_and_save_npz(path, output_path, suffix=".npy")
        print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    main()
