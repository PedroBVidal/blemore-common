import os
import numpy as np
from tqdm import tqdm
import sys
import numpy as np
import re
import pandas as pd



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



def get_video_info(video_name=''):
    video_type = 'single' if not 'mix' in video_name else 'mix'
    info = {'type': video_type}
    video_name_split = video_name.split('_')
    if video_type == 'single':
        actor, emotion, intensity, version = video_name_split
        info['actor']     = actor
        info['emotion']   = emotion
        info['intensity'] = intensity
        info['version']   = version
    elif video_type == 'mix':
        actor, _, emotion1, emotion2, perc1, perc2, version = video_name_split
        info['actor']     = actor
        info['emotion']   = emotion1 + '_' + emotion2
        info['intensity'] = perc1    + '_' + perc2
        info['version']   = version
    return info

def aggregate_bfm_transfer_expression_and_save_npz(source_dir, output_path, suffix=".npy", df=None):
    all_features = []
    all_filenames = []
    all_subdirs_videos_paths = get_imediate_subdirs_paths(source_dir)
    
    # Load default BFM features
    print('Loading default BFM features...')
    for idx_subdir_video, path_subdir_video in enumerate(all_subdirs_videos_paths):
        video_name = os.path.basename(path_subdir_video)
        print(f"{idx_subdir_video}/{len(all_subdirs_videos_paths)} - {path_subdir_video}                      ", end='\r')
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


    # Transfer expressions
    print('Transfering expressions through BFM features...')
    for idx_subdir_video_base, path_subdir_video_base in enumerate(all_subdirs_videos_paths):
        video_name_base = os.path.basename(path_subdir_video_base)
        video_info_base = get_video_info(video_name_base)
        fold_video_base = df[df['filename']==video_name_base]['fold'].tolist()[0]

        if video_info_base['type']=='single' and video_info_base['emotion']=='neu' and video_info_base['intensity']=='sit1':
            # print(f"{idx_subdir_video_base}/{len(all_subdirs_videos_paths)} - '{video_name_base}'")
            for idx_subdir_video_ref, path_subdir_video_ref in enumerate(all_subdirs_videos_paths):
                video_name_ref = os.path.basename(path_subdir_video_ref)
                video_info_ref = get_video_info(video_name_ref)
                fold_video_ref = df[df['filename']==video_name_ref]['fold'].tolist()[0]
                if idx_subdir_video_base != idx_subdir_video_ref and \
                   fold_video_base == fold_video_ref and \
                   video_info_base['actor']   != video_info_ref['actor'] and \
                   video_info_base['emotion'] != video_info_ref['emotion']:
                    video_name_transf = video_name_ref.replace(suffix, "").replace(video_info_ref['actor'],video_info_base['actor'])
                    print(f"{idx_subdir_video_base}/{len(all_subdirs_videos_paths)} video_name_base: '{video_name_base}'    {idx_subdir_video_ref}/{len(all_subdirs_videos_paths)} video_name_ref: {video_name_ref}    video_name_transf: {video_name_transf}                        ", end='\r')
                    # print(f"     {idx_subdir_video_ref}/{len(all_subdirs_videos_paths)} - '{video_name_ref}'")
                    # print("path_subdir_video_base:", path_subdir_video_base)
                    
                    files_paths_base = get_all_files_any_depth(path_subdir_video_base, extension=".npy")
                    first_frame_feature_base = np.load(files_paths_base[0])
                    # print('files_paths_base[0]:', files_paths_base[0])
                    # print('first_frame_feature.shape:', first_frame_feature.shape)
                    # sys.exit(0)

                    # id_coeffs = coeffs[:, :80]         # face identity
                    # exp_coeffs = coeffs[:, 80: 144]    # face expression
                    # tex_coeffs = coeffs[:, 144: 224]
                    # angles = coeffs[:, 224: 227]       # face rotation
                    # gammas = coeffs[:, 227: 254]
                    # translations = coeffs[:, 254:]     # face translation
                    # return {
                    #     'id': id_coeffs,
                    #     'exp': exp_coeffs,
                    #     'tex': tex_coeffs,
                    #     'angle': angles,
                    #     'gamma': gammas,
                    #     'trans': translations
                    # }
                    files_paths_ref = get_all_files_any_depth(path_subdir_video_ref, extension=".npy")
                    first_frame_feature_ref = np.load(files_paths_ref[0])
                    video_frames_bfm_ref = np.zeros((len(files_paths_ref), first_frame_feature_ref.shape[-1]), dtype=np.float32)
                    for idx_file_ref, path_file_ref in enumerate(files_paths_ref):
                        x_ref = np.load(path_file_ref)
                        x_ref[:,:80]     = first_frame_feature_base[:,:80]        # face identity
                        x_ref[:,144:224] = first_frame_feature_base[:,144:224]    # texture
                        video_frames_bfm_ref[idx_file_ref,:] = x_ref

                    agg = np.concatenate([
                        video_frames_bfm_ref.mean(axis=0),
                        video_frames_bfm_ref.std(axis=0),
                        np.percentile(video_frames_bfm_ref, 10, axis=0),
                        np.percentile(video_frames_bfm_ref, 25, axis=0),
                        np.percentile(video_frames_bfm_ref, 50, axis=0),  # median
                        np.percentile(video_frames_bfm_ref, 75, axis=0),
                        np.percentile(video_frames_bfm_ref, 90, axis=0),
                    ])

                    # print('video_name_transf:', video_name_transf)
                    # sys.exit(0)
                    all_features.append(agg)
                    all_filenames.append(video_name_transf)

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
    # base_static_dir = "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_test_data"

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
        # "bfm": "/home/pbqv20/BlEmoRe_backup/data_frames_HRN_3D_reconstruction/train/all_parts/",
        # "bfm": "/home/pbqv20/BlEmoRe_backup/data_frames_HRN_3D_reconstruction/test/test_videos/",
        "bfm_transfer_exp": "/home/pbqv20/BlEmoRe_backup/data_frames_HRN_3D_reconstruction/train/all_parts/",

        # "imagebind_wavlm": ["/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/ImageBind_train/",
        #                     "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/WavLM_large_train"]
        # "videomae_hubert": ["/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/VideoMAEv2_train/",
        #                     "/home/pbqv20/BlEmoRe_backup/feat/pre_extracted_train_data/HuBERT_large_train/"]
    }


    '''
    train_df:                 filename  video_id  gender  mix emotion_1  version  intensity_level  situation emotion_2  emotion_1_salience  emotion_2_salience  fold                                                                                                                                               
    0               A102_ang_int1_ver1      A102       f    0       ang        1              1.0        NaN       NaN                 NaN                 NaN     1                                                                                                                                                         
    1               A102_ang_int2_ver1      A102       f    0       ang        1              2.0        NaN       NaN                 NaN                 NaN     1                                                                                                                                                         
    2               A102_ang_int3_ver1      A102       f    0       ang        1              3.0        NaN       NaN                 NaN                 NaN     1                                                                                                                                                         
    3               A102_ang_int4_ver1      A102       f    0       ang        1              4.0        NaN       NaN                 NaN                 NaN     1                                                                                                                                                         
    4              A102_disg_int1_ver1      A102       f    0      disg        1              1.0        NaN       NaN                 NaN                 NaN     1             
    '''
    train_metadata_path = "/home/pbqv20/BlEmoRe_backup/train_metadata.csv"
    print(f"Loading train metadata: \'{train_metadata_path}\'")
    train_df = pd.read_csv(train_metadata_path)


    for encoder, path in encoding_paths.items():
        print(f"Processing {encoder} from {path}...")
        if type(path) is str:
            output_path = os.path.join(base_static_dir, f"{encoder}_static_features.npz")
            if encoder == "bfm":
                aggregate_bfm_and_save_npz(path, output_path, suffix=".npy")
            elif encoder == "bfm_transfer_exp":
                aggregate_bfm_transfer_expression_and_save_npz(path, output_path, suffix=".npy", df=train_df)
            else:
                aggregate_and_save_npz(path, output_path, suffix=".npy")
        elif type(path) is list:
            output_path = os.path.join(base_static_dir, f"{encoder}_fused.npz")
            aggregate_fuse_and_save_npz(path, output_path, suffix=".npy")
        print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    main()
