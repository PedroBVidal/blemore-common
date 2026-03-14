import os

def get_available_encoders():
    """Central source for active encoders and model types."""
    vision_encoders = []
    audio_encoders = []
    encoder_fusions = ["videomae_hubert"]
    
    encoders = vision_encoders + audio_encoders + encoder_fusions
    model_types = ["MLP_512"]
    
    return encoders, model_types

def get_encoding_paths(data_folder, mode="train"):
    """
    Returns the dictionary of paths. 
    Switches between pre_extracted_train_data and pre_extracted_test_data.
    """
    subfolder = f"pre_extracted_{mode}_data"
    
    # We use a helper to join the common base
    base = os.path.join(data_folder, "feat", subfolder)
    
    # Nested folder helper (for things like openface/fused)
    static_base = os.path.join(base, "encoded_videos/static_data")

    return {
        # Vision
        "imagebind": os.path.join(base, "imagebind_static_features.npz"),
        "openface": os.path.join(static_base, "openface_static_features.npz"),
        "clip": os.path.join(static_base, "clip_static_features.npz"),
        "videoswintransformer": os.path.join(static_base, "videoswintransformer_static_features.npz"),
        "videomae": os.path.join(static_base, "videomae_static_features.npz"),

        # Audio
        "wavlm": os.path.join(static_base, "wavlm_static_features.npz"),
        "hubert": os.path.join(static_base, "hubert_static_features.npz"),

        # Fused
        "imagebind_wavlm": os.path.join(base, "imagebind_wavlm_fused.npz"),
        "imagebind_hubert": os.path.join(base, "imagebind_hubert_fused.npz"),
        "videomae_wavlm": os.path.join(static_base, "fused/videomae_wavlm_fused.npz"),
        "videomae_hubert": os.path.join(base, "videomae_hubert_fused.npz"),
        
        # Multimodal
        "hicmae": os.path.join(static_base, "hicmae_static_features.npz"),
    }