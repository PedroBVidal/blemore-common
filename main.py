<<<<<<< HEAD
from json import encoder
import os
=======
import os, sys
>>>>>>> origin/biesseck
import json
from collections import Counter

import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

<<<<<<< HEAD
from model.model import ConfigurableLinearNN
=======
from model.model import ConfigurableLinearNN, Backbone_with_ConfigurableLinearNN
from model.iresnet_1d import iresnet18_1d, iresnet50_1d
from model.iresnet_2d import get_model
from model.resnet_lstm import ResNetLSTM
from model.lstm import SequenceLSTM, LSTM_Attention
>>>>>>> origin/biesseck
from model.post_process import get_top_2_predictions, probs2dict
from trainer.trainer import Trainer
from utils.create_soft_labels import create_labels
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
<<<<<<< HEAD
from utils.set_splitting import prepare_split_2d, prepare_train_2d, prepare_test_2d, get_validation_split
from utils.get_encoding_paths import get_available_encoders, get_encoding_paths
=======
from utils.set_splitting import prepare_split_2d, get_validation_split
>>>>>>> origin/biesseck

# --- Global Settings ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default='', help='Information to be added to output folder')
    args = parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
<<<<<<< HEAD
    "batch_size": 32,
    "learning_rate": 5e-6,
    # "num_epochs": 200,
    "num_epochs": 400,
    "weight_decay": 1e-3,
=======
    # "batch_size": 32,
    "batch_size": 128,
    "learning_rate": 5e-6,
    # "learning_rate": 1e-7,
    "num_epochs": 400,
    "weight_decay": 1e-3,
    # "weight_decay": 1e-6,
>>>>>>> origin/biesseck
}

# data_folder = "/home/tim/Work/quantum/data/blemore/"
data_folder = "/home/pbqv20/BlEmoRe_backup"


<<<<<<< HEAD

train_metadata_path = os.path.join(data_folder, "train_metadata.csv")
# train_metadata_path = os.path.join(data_folder, "train_metadata_balanced.csv")

test_metadata_path = os.path.join(data_folder, "test_metadata.csv")


#now encoding paths are saved on utils/get_encoding_pths.py 
encoding_paths = {
    # vision
    # "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
    #"imagebind": os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_static_features.npz"),
    # "imagebind": os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_STACK_features.npz"),
    # "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
    # "videoswintransformer": os.path.join(data_folder,
    #                                      "encoded_videos/static_data/videoswintransformer_static_features.npz"),
    # "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),
=======
# train_metadata_path = os.path.join(data_folder, "train_metadata_balanced.csv")               # all actors in all folds
# train_metadata_path = os.path.join(data_folder, "train_metadata_ONLY_SINGLE_EMOTIONS.csv")
# train_metadata_path = os.path.join(data_folder, "train_metadata_ONLY_BLENDED_EMOTIONS.csv")
train_metadata_path = os.path.join(data_folder, "train_metadata.csv")                          # default
test_metadata_path = os.path.join(data_folder, "test_metadata.csv")                            # default


encoding_paths = {
    # vision
    # "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
    "imagebind": os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_static_features.npz"),
    # "imagebind": os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_STACK_features.npz"),
    
    "seq_imagebind_sequence=32_padding=False":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=32_padding=False.npz"),
    "seq_imagebind_sequence=64_padding=True":   os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=64_padding=True.npz"),
    "seq_imagebind_sequence=128_padding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=128_padding=True.npz"),
    "seq_imagebind_sequence=256_padding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=256_padding=True.npz"),
    "seq_imagebind_sequence=512_padding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=512_padding=True.npz"),
    "seq_imagebind_sequence=1024_padding=True": os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=1024_padding=True.npz"),
    
    "seq_imagebind_sequence=128_innerpadding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=128_innerpadding=True.npz"),
    "seq_imagebind_sequence=256_innerpadding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=256_innerpadding=True.npz"),
    "seq_imagebind_sequence=512_innerpadding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=512_innerpadding=True.npz"),
    "seq_imagebind_sequence=1024_innerpadding=True":  os.path.join(data_folder, "feat/pre_extracted_train_data/seq_imagebind_SEQUENCE_features_sequence=1024_innerpadding=True.npz"),
    
    # "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
    # "videoswintransformer": os.path.join(data_folder, "encoded_videos/static_data/videoswintransformer_static_features.npz"),
    # "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),
    "rawimgs112x112":     os.path.join(data_folder, "feat/pre_extracted_train_data/rawimgs112x112_STACK_features_subsampling=5.npz"),
    "seq_rawimgs112x112": os.path.join(data_folder, "feat/pre_extracted_train_data/seq_rawimgs112x112_SEQUENCE_features_sequence=32.npz"),
    "bfm": os.path.join(data_folder, "feat/pre_extracted_train_data/bfm_static_features.npz"),
>>>>>>> origin/biesseck

    # audio
    # "wavlm": os.path.join(data_folder, "encoded_videos/static_data/wavlm_static_features.npz"),
    # "hubert": os.path.join(data_folder, "encoded_videos/static_data/hubert_static_features.npz"),

    # fused
    # "imagebind_wavlm": os.path.join(data_folder, "encoded_videos/static_data/fused/imagebind_wavlm_fused.npz"),
    # "imagebind_hubert": os.path.join(data_folder, "encoded_videos/static_data/fused/imagebind_hubert_fused.npz"),
    # "videomae_wavlm": os.path.join(data_folder, "encoded_videos/static_data/fused/videomae_wavlm_fused.npz"),
    # "videomae_hubert": os.path.join(data_folder, "encoded_videos/static_data/fused/videomae_hubert_fused.npz"),

    # multimodal
    # "hicmae": os.path.join(data_folder, "encoded_videos/static_data/hicmae_static_features.npz"),
}


def select_model(model_type, input_dim, output_dim):
    if model_type == "Linear":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=0)
    elif model_type == "MLP_256":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=256)
    elif model_type == "MLP_512":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
<<<<<<< HEAD
=======
    elif model_type == "resnet18_1d":
        backbone = iresnet18_1d(input_channels=1, num_features=128)
        class_model = Backbone_with_ConfigurableLinearNN(backbone, input_dim=128, output_dim=output_dim, model_type= model_type, n_layers=0)
        return class_model
    elif model_type == "resnet18_2d":
        network = "r18"
        fp16 = True
        num_features=512
        backbone = get_model(network, dropout=0.0, fp16=fp16, num_features=num_features)
        class_model = Backbone_with_ConfigurableLinearNN(backbone, input_dim=num_features, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
        return class_model
    elif model_type == "r50_lstm":
        hidden_size = 512
        backbone = ResNetLSTM(hidden_size=hidden_size, num_layers=2)
        class_model = Backbone_with_ConfigurableLinearNN(backbone, input_dim=hidden_size, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
        return class_model
    elif model_type == "lstm":
        # hidden_size = 512
        hidden_size = 4096
        num_layers = 1
        backbone = SequenceLSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers)
        class_model = Backbone_with_ConfigurableLinearNN(backbone, input_dim=hidden_size, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
        return class_model
    elif model_type == "lstm_att":
        # hidden_size = 256
        # hidden_size = 512
        # hidden_size = 2048
        hidden_size = 4096
        num_layers = 1
        backbone = LSTM_Attention(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers)
        class_model = Backbone_with_ConfigurableLinearNN(backbone, input_dim=hidden_size, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
        return class_model
>>>>>>> origin/biesseck
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_fold(train_dataset, val_dataset, model_type, log_dir, save_prefix):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

<<<<<<< HEAD
    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
=======
    # model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    model = select_model(model_type, train_dataset.X.shape[-1], train_dataset.output_dim)
>>>>>>> origin/biesseck
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=hparams["num_epochs"],
                      valid_data_loader=val_loader, subsample_aggregation=False)

    writer = SummaryWriter(log_dir=log_dir)
    best_epoch, best_model_path = trainer.train(writer=writer, save_prefix=save_prefix)
    writer.close()
    return best_epoch, best_model_path


def train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha, beta, encoder):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=100,
                      subsample_aggregation=False)

    trainer.train()

    return evaluate_model(model, test_loader, alpha, beta, encoder)


<<<<<<< HEAD
def evaluate_model(model, test_loader, alpha, beta, encoder,isTest=False):
=======
def evaluate_model(model, test_loader, alpha, beta, encoder):
>>>>>>> origin/biesseck
    all_probs = []
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
<<<<<<< HEAD
            print(data.shape)
=======
>>>>>>> origin/biesseck
            data = data.to(device)
            probs, _, _ = model(data)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    top_2_probs = get_top_2_predictions(all_probs)
    test_filenames = test_loader.dataset.filenames

<<<<<<< HEAD
    final_preds = probs2dict(top_2_probs, test_filenames, alpha, beta,isTest)
=======
    final_preds = probs2dict(top_2_probs, test_filenames, alpha, beta)
>>>>>>> origin/biesseck

    with open("data/{}_test_predictions.json".format(encoder), "w") as f:
        json.dump(final_preds, f, indent=4)

<<<<<<< HEAD
    acc_presence,acc_salience = None, None
    if (isTest == False):
        acc_presence = acc_presence_total(final_preds)
        acc_salience = acc_salience_total(final_preds)
     
    return acc_presence, acc_salience

def run_validation(train_df, train_labels, encoders_list, model_types, args,data_folder):
    # folds = [0, 1, 2, 3, 4]
    folds = [0, 1]

    # Get the dictionary of actual files for TRAINING mode
    all_encoding_paths = get_encoding_paths(data_folder, mode="train")

    for encoder in encoders_list:
        encoding_path = all_encoding_paths[encoder]
        summary_rows = []
=======
    acc_presence = acc_presence_total(final_preds)
    acc_salience = acc_salience_total(final_preds)

    return acc_presence, acc_salience

def run_validation(train_df, train_labels, encoders, model_types, args):
    # folds = [0, 1, 2, 3, 4]
    folds = [0]

    summary_rows = []
    for encoder in encoders:
        encoding_path = encoding_paths[encoder]

>>>>>>> origin/biesseck
        for model_type in model_types:
            fold_results = []
            for fold_id in folds:
                print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}")
                (train_files, train_labels_fold), (val_files, val_labels) = get_validation_split(train_df, train_labels, fold_id)
                train_dataset, val_dataset = prepare_split_2d(train_files, train_labels_fold, val_files, val_labels, encoding_path)

                log_dir = f"runs/{encoder}_{model_type}_fold{fold_id}"
                if args.annotation:
                    log_dir += f'_annotation={args.annotation}'
                save_prefix = f"{encoder}_{model_type}_fold{fold_id}"
                best_epoch, _ = train_one_fold(train_dataset, val_dataset, model_type, log_dir, save_prefix)

                best_epoch.update({"encoder": encoder, "model": model_type, "fold": fold_id})
                summary_rows.append(best_epoch)
                fold_results.append(best_epoch)

    # Save validation results
    summary_df = pd.DataFrame(summary_rows)
<<<<<<< HEAD
    summary_df.to_csv(f"data/validation_summary_{encoder}_{model_type}.csv", index=False)
=======
    summary_df.to_csv("validation_summary.csv", index=False)
>>>>>>> origin/biesseck
    print("\nValidation Summary:")
    print(summary_df)

    return summary_df


<<<<<<< HEAD
def run_test(train_df, train_labels, test_df, encoders, model_types, use_best_model_from_val=True, use_fold_id=None):
    test_summary_rows = []

    # 1. Pre-generate all paths to avoid repeated function calls
    train_paths = get_encoding_paths(data_folder, mode="train")
    test_paths = get_encoding_paths(data_folder, mode="test")

    for encoder in encoders:
        # Get the specific .npz files for this encoder
        path_train = train_paths[encoder]
        path_test = test_paths[encoder]

        for model_type in model_types:
            # Load validation summary to find best alpha/beta/fold
            summary_path = f"data/validation_summary_{encoder}_{model_type}.csv"
            if not os.path.exists(summary_path):
                print(f"Skip: {summary_path} not found.")
                continue
            
            summary_df = pd.read_csv(summary_path)
=======
def run_test(train_df, train_labels, test_df, test_labels, encoders, model_types, use_best_model_from_val=True, use_fold_id=None):
    test_summary_rows = []

    # Load validation summary
    summary_df = pd.read_csv("data/validation_summary_hicmae.csv")

    for encoder in encoders:
        encoding_path = encoding_paths[encoder]

        for model_type in model_types:
>>>>>>> origin/biesseck
            # Filter for encoder and model type
            fold_df = summary_df[(summary_df["encoder"] == encoder) & (summary_df["model"] == model_type)]

            # Select alpha and beta from the best fold
            best_row = fold_df.loc[
                (0.5 * fold_df["best_acc_presence"] + 0.5 * fold_df["best_acc_salience"]).idxmax()
            ]
            alpha_best = best_row["best_alpha"]
            beta_best = best_row["best_beta"]
<<<<<<< HEAD
            fold_id = use_fold_id if use_fold_id is not None else int(best_row["fold"])

            print(f"\n--- Testing {encoder} | {model_type} ---")
            print(f"Alpha: {alpha_best:.4f}, Beta: {beta_best:.4f}, Fold: {fold_id}")
=======
            fold_id = best_row["fold"]

            print(f"Selected alpha: {alpha_best:.4f}, beta: {beta_best:.4f} for encoder={encoder}, model={model_type}")
>>>>>>> origin/biesseck

            # Train on full train set and evaluate on test set
            train_files = train_df.filename.tolist()
            test_files = test_df.filename.tolist()
<<<<<<< HEAD
            train_dataset, fitted_scaler = prepare_train_2d(train_files, train_labels, path_train)

            test_dataset = prepare_test_2d(test_files, path_test,fitted_scaler)

            if use_best_model_from_val:

=======
            train_dataset, test_dataset = prepare_split_2d(train_files, train_labels, test_files, test_labels, encoding_path)

            if use_best_model_from_val:

                if use_fold_id is not None:
                    fold_id = use_fold_id

>>>>>>> origin/biesseck
                # Use best model from validation
                best_model_path = f"checkpoints/{encoder}_{model_type}_fold{fold_id}_best.pth"
                print(f"Loading model from {best_model_path}")

                checkpoint = torch.load(best_model_path, map_location=device)
                model = select_model(checkpoint['model_type'], checkpoint['input_dim'], checkpoint['output_dim'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

<<<<<<< HEAD
                #generate json with pred for codabech submission
                acc_presence, acc_salience = evaluate_model(model, test_loader, alpha_best, beta_best, encoder,isTest=True)

            else:
                acc_presence, acc_salience = train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha_best, beta_best, encoder)

            if acc_presence is not None and acc_salience is not None:
                print(f"Results -> Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")
=======
                acc_presence, acc_salience = evaluate_model(model, test_loader, alpha_best, beta_best, encoder)
            else:
                acc_presence, acc_salience = train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha_best, beta_best, encoder)

            print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")
            print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")
>>>>>>> origin/biesseck

            # Save test results
            test_summary_rows.append({
                "encoder": encoder,
                "model": model_type,
                "alpha": alpha_best,
                "beta": beta_best,
                "test_acc_presence": acc_presence,
                "test_acc_salience": acc_salience,
            })

    # Save test results
    test_summary_df = pd.DataFrame(test_summary_rows)
    test_summary_df.to_csv("test_summary.csv", index=False)
    print("\nTest Summary:")
    print(test_summary_df)

    # Encoder/model-averaged test results
    print("\nAveraged Test Results:")
    print(test_summary_df.groupby(["encoder", "model"])[["test_acc_presence", "test_acc_salience"]].mean())

<<<<<<< HEAD
'''
def main(config, do_val=True, do_test=False, args=None):

    # vision_encoders = ["imagebind", "videomae", "videoswintransformer", "openface", "clip"]
    #vision_encoders = ["imagebind"]
    
    # audio_encoders = ["wavlm", "hubert"]
    #audio_encoders = []

    # encoder_fusions = ["imagebind_wavlm", "imagebind_hubert", "videomae_wavlm", "videomae_hubert"]
    #encoder_fusions = []

    #encoders = vision_encoders + audio_encoders + encoder_fusions

    # model_types = ["Linear", "MLP_256", "MLP_512"]
    #model_types = ["MLP_256"]

    train_df = pd.read_csv(train_metadata_path)
    train_labels = create_labels(train_df.to_dict(orient="records"))

    if do_val:
        print(f"Loading train protocol \'{train_metadata_path}\'")
        run_validation(train_df, train_labels, encoders, model_types, args)

    if do_test:
        print(f"Loading test protocol \'{train_metadata_path}\'")
        test_df = pd.read_csv(test_metadata_path)
        run_test(train_df, train_labels, test_df, encoders, model_types, use_best_model_from_val=True)
'''

def main(data_folder, do_val=False, do_test=True, args=None):

    # 1. Get the lists (Vision/Audio/Fusion combined)
    encoders_to_run, model_types = get_available_encoders()

    train_metadata_path = os.path.join(data_folder, "train_metadata.csv")
    
    train_df = pd.read_csv(train_metadata_path)
    train_labels = create_labels(train_df.to_dict(orient="records"))

    if do_val:
        run_validation(train_df, train_labels, encoders_to_run, model_types, args, data_folder)

    if do_test:
        test_df = pd.read_csv(os.path.join(data_folder, "test_metadata.csv"))
        run_test(train_df, train_labels, test_df, encoders_to_run, model_types, data_folder)

if __name__ == "__main__":
    args = parse_args()
    main(data_folder, do_val=False, do_test=True, args=args)
=======

def main(do_val=True, do_test=False, args=None):
    # vision_encoders = ["imagebind", "videomae", "videoswintransformer", "openface", "clip"]
    # vision_encoders = ["imagebind"]
    
    # vision_encoders = ["seq_imagebind_sequence=32_padding=False"]
    # vision_encoders = ["seq_imagebind_sequence=64_padding=True"]
    # vision_encoders = ["seq_imagebind_sequence=128_padding=True"]
    # vision_encoders = ["seq_imagebind_sequence=256_padding=True"]
    # vision_encoders = ["seq_imagebind_sequence=512_padding=True"]
    # vision_encoders = ["seq_imagebind_sequence=1024_padding=True"]
    
    # vision_encoders = ["seq_imagebind_sequence=128_innerpadding=True"]
    # vision_encoders = ["seq_imagebind_sequence=256_innerpadding=True"]
    # vision_encoders = ["seq_imagebind_sequence=512_innerpadding=True"]
    # vision_encoders = ["seq_imagebind_sequence=1024_innerpadding=True"]
    # vision_encoders = ["rawimgs112x112"]
    # vision_encoders = ["seq_rawimgs112x112"]

    vision_encoders = ["bfm"]


    # audio_encoders = ["wavlm", "hubert"]
    audio_encoders = []


    # encoder_fusions = ["imagebind_wavlm", "imagebind_hubert", "videomae_wavlm", "videomae_hubert"]
    encoder_fusions = []

    encoders = vision_encoders + audio_encoders + encoder_fusions

    # model_types = ["Linear", "MLP_256", "MLP_512"]
    model_types = ["MLP_512"]
    # model_types = ["resnet18_1d"]
    # model_types = ["resnet18_2d"]
    # model_types = ["r50_lstm"]
    # model_types = ["lstm"]
    # model_types = ["lstm_att"]

    if do_val:
        train_df = pd.read_csv(train_metadata_path)
        train_labels = create_labels(train_df.to_dict(orient="records"))

        run_validation(train_df, train_labels, encoders, model_types, args)

    if do_test:
        test_df = pd.read_csv(test_metadata_path)
        test_labels = create_labels(test_df.to_dict(orient="records"))

        run_test(train_df, train_labels, test_df, test_labels, encoders, model_types, use_best_model_from_val=False)

if __name__ == "__main__":
    args = parse_args()
    main(do_val=True, do_test=False, args=args)
>>>>>>> origin/biesseck
