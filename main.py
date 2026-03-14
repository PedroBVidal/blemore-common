import os, sys
import json
from collections import Counter

import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model import ConfigurableLinearNN
from model.post_process import get_top_2_predictions, probs2dict
from trainer.trainer import Trainer
from utils.create_soft_labels import create_labels
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
from utils.set_splitting import prepare_split_2d, get_validation_split, prepare_test_split_2d

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
    parser.add_argument('--test-model', type=str, default='', help='Path of weights file (.pt, .pth, etc.)')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    args = parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    # "batch_size": 32,
    "batch_size": 256,
    "learning_rate": 5e-6,
    # "num_epochs": 200,
    "num_epochs": 400,
    "weight_decay": 1e-3,
}

# data_folder = "/home/tim/Work/quantum/data/blemore/"
data_folder = "/home/pbqv20/BlEmoRe_backup"



# train_metadata_path = os.path.join(data_folder, "train_metadata_balanced.csv")   # all actors in all folds
# train_metadata_path = os.path.join(data_folder, "train_metadata_ONLY_SINGLE_EMOTIONS.csv")
# train_metadata_path = os.path.join(data_folder, "train_metadata_ONLY_BLENDED_EMOTIONS.csv")
train_metadata_path = os.path.join(data_folder, "train_metadata.csv")              # default
test_metadata_path = os.path.join(data_folder, "test_metadata.csv")                # default



encoding_paths = {
    # vision
    # "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
    "imagebind":             os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_static_features.npz"),
    "imagebind11statistics": os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_static_features_11statistics.npz"),
    # "imagebind": os.path.join(data_folder, "feat/pre_extracted_train_data/imagebind_STACK_features.npz"),
    # "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
    # "videoswintransformer": os.path.join(data_folder,
    #                                      "encoded_videos/static_data/videoswintransformer_static_features.npz"),
    # "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),
    "bfm":                     os.path.join(data_folder, "feat/pre_extracted_train_data/bfm_static_features.npz"),
    "bfm_transfer_single_exp": os.path.join(data_folder, "feat/pre_extracted_train_data/bfm_transfer_exp_static_features_TRANSFER_ONLY_SINGLE_EMOTIONS.npz"),
    "bfm_transfer_all_exp":    os.path.join(data_folder, "feat/pre_extracted_train_data/bfm_transfer_exp_static_features.npz"),

    # audio
    # "wavlm": os.path.join(data_folder, "encoded_videos/static_data/wavlm_static_features.npz"),
    # "hubert": os.path.join(data_folder, "encoded_videos/static_data/hubert_static_features.npz"),

    # fused
    # "imagebind_wavlm": os.path.join(data_folder, "encoded_videos/static_data/fused/imagebind_wavlm_fused.npz"),
    # "imagebind_hubert": os.path.join(data_folder, "encoded_videos/static_data/fused/imagebind_hubert_fused.npz"),
    # "videomae_wavlm": os.path.join(data_folder, "encoded_videos/static_data/fused/videomae_wavlm_fused.npz"),
    "videomae_hubert": os.path.join(data_folder, "feat/pre_extracted_train_data/videomae_hubert_fused.npz"),

    # multimodal
    # "hicmae": os.path.join(data_folder, "encoded_videos/static_data/hicmae_static_features.npz"),
}


test_encoding_paths = {
    "videomae_hubert": os.path.join(data_folder, "feat/pre_extracted_test_data/videomae_hubert_fused.npz"),
}


def select_model(model_type, input_dim, output_dim):
    if model_type == "Linear":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=0)
    elif model_type == "MLP_256":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=256)
    elif model_type == "MLP_512":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_fold(train_dataset, val_dataset, model_type, log_dir, save_prefix, fold_id):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=hparams["num_epochs"],
                      valid_data_loader=val_loader, subsample_aggregation=False)

    writer = SummaryWriter(log_dir=log_dir)
    best_epoch, best_model_path = trainer.train(writer=writer, save_prefix=save_prefix, fold_id=fold_id)
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


def evaluate_model(model, test_loader, alpha, beta, encoder):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            probs, _, _ = model(data)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    top_2_probs = get_top_2_predictions(all_probs)
    test_filenames = test_loader.dataset.filenames

    final_preds = probs2dict(top_2_probs, test_filenames, alpha, beta)

    with open("data/{}_test_predictions.json".format(encoder), "w") as f:
        json.dump(final_preds, f, indent=4)

    acc_presence = acc_presence_total(final_preds)
    acc_salience = acc_salience_total(final_preds)

    return acc_presence, acc_salience

def run_validation(train_df, train_labels, encoders, model_types, args):
    # folds = [0, 1, 2, 3, 4]
    # folds = [0, 1]
    folds = [0]

    summary_rows = []
    for encoder in encoders:
        encoding_path = encoding_paths[encoder]

        for model_type in model_types:
            fold_results = []
            for fold_id in folds:
                print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}")
                (train_files, train_labels_fold), (val_files, val_labels) = get_validation_split(train_df, train_labels, fold_id)
                train_dataset, val_dataset = prepare_split_2d(train_files, train_labels_fold, val_files, val_labels, encoding_path)

                # save_prefix = f"{encoder}_{model_type}_fold{fold_id}"
                save_prefix = f"{encoder}_{model_type}"
                if args.annotation:
                    save_prefix += f'_annotation={args.annotation}'
                log_dir = f"runs/{save_prefix}/{save_prefix}_fold{fold_id}"
                best_epoch, _ = train_one_fold(train_dataset, val_dataset, model_type, log_dir, save_prefix, fold_id)

                best_epoch.update({"encoder": encoder, "model": model_type, "fold": fold_id})
                summary_rows.append(best_epoch)
                fold_results.append(best_epoch)

    # Save validation results
    summary_df = pd.DataFrame(summary_rows)
    # summary_df.to_csv("validation_summary.csv", index=False)
    summary_df.to_csv(os.path.join(os.path.dirname(log_dir), f"validation_summary_folds={str(folds).replace(' ','')}.csv"), index=False)
    print("\nValidation Summary:")
    print(summary_df)

    return summary_df


def run_test(train_df, train_labels, test_df, test_labels, encoders, model_types, use_best_model_from_val=True, use_fold_id=None):
    test_summary_rows = []

    # Load validation summary
    summary_df = pd.read_csv("data/validation_summary_hicmae.csv")

    for encoder in encoders:
        encoding_path = encoding_paths[encoder]

        for model_type in model_types:
            # Filter for encoder and model type
            fold_df = summary_df[(summary_df["encoder"] == encoder) & (summary_df["model"] == model_type)]

            # Select alpha and beta from the best fold
            best_row = fold_df.loc[
                (0.5 * fold_df["best_acc_presence"] + 0.5 * fold_df["best_acc_salience"]).idxmax()
            ]
            alpha_best = best_row["best_alpha"]
            beta_best = best_row["best_beta"]
            fold_id = best_row["fold"]

            print(f"Selected alpha: {alpha_best:.4f}, beta: {beta_best:.4f} for encoder={encoder}, model={model_type}")

            # Train on full train set and evaluate on test set
            train_files = train_df.filename.tolist()
            test_files = test_df.filename.tolist()
            train_dataset, test_dataset = prepare_split_2d(train_files, train_labels, test_files, test_labels, encoding_path)

            if use_best_model_from_val:

                if use_fold_id is not None:
                    fold_id = use_fold_id

                # Use best model from validation
                best_model_path = f"checkpoints/{encoder}_{model_type}_fold{fold_id}_best.pth"
                print(f"Loading model from {best_model_path}")

                checkpoint = torch.load(best_model_path, map_location=device)
                model = select_model(checkpoint['model_type'], checkpoint['input_dim'], checkpoint['output_dim'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

                acc_presence, acc_salience = evaluate_model(model, test_loader, alpha_best, beta_best, encoder)
            else:
                acc_presence, acc_salience = train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha_best, beta_best, encoder)

            print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")
            print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")

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




def get_test_preds(train_df, train_labels, test_df, encoders, model_types, use_best_model_from_val=True, use_fold_id=None, args=None):
    # test_summary_rows = []

    # Load validation summary
    # summary_df = pd.read_csv("data/validation_summary_hicmae.csv")
    # summary_df = pd.read_csv("/home/bjgbiesseck/GitHub/PedroBVidal_blemore-common/runs/videomae_hubert_MLP_512_fold0_annotation=BASELINE-ON-videomae_hubert/validation_summary.csv")

    for encoder in encoders:
        encoding_path = encoding_paths[encoder]
        test_encoding_path = test_encoding_paths[encoder]

        for model_type in model_types:
            # Filter for encoder and model type
            # fold_df = summary_df[(summary_df["encoder"] == encoder) & (summary_df["model"] == model_type)]

            # Select alpha and beta from the best fold
            # best_row = fold_df.loc[
            #     (0.5 * fold_df["best_acc_presence"] + 0.5 * fold_df["best_acc_salience"]).idxmax()
            # ]
            # alpha_best = best_row["best_alpha"]
            # beta_best = best_row["best_beta"]
            # fold_id = best_row["fold"]
            # print(f"Selected alpha: {alpha_best:.4f}, beta: {beta_best:.4f} for encoder={encoder}, model={model_type}")

            # Train on full train set and evaluate on test set
            train_files = train_df.filename.tolist()
            test_files = test_df.filename.tolist()
            # train_dataset, test_dataset = prepare_split_2d(train_files, train_labels, test_files, test_labels, encoding_path)
            train_dataset, test_dataset = prepare_test_split_2d(train_files, train_labels, encoding_path, test_files, test_encoding_path)
            # print('test_dataset.X.shape:', test_dataset.X.shape)
            # sys.exit(0)

            if use_best_model_from_val:

                if use_fold_id is not None:
                    fold_id = use_fold_id

                # Use best model from validation
                # best_model_path = f"checkpoints/{encoder}_{model_type}_fold{fold_id}_best.pth"
                # best_model_path = f"checkpoints/{encoder}_{model_type}_fold{int(fold_id)}_best.pth"
                best_model_path = args.test_model
                print(f"Loading model from {best_model_path}")

                checkpoint = torch.load(best_model_path, map_location=device)
                model = select_model(checkpoint['model_type'], checkpoint['input_dim'], checkpoint['output_dim'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

                # acc_presence, acc_salience = evaluate_model(model, test_loader, alpha_best, beta_best, encoder)
                all_probs = []
                model.eval()
                with torch.no_grad():
                    print(f"Performing test predictions (alpha={args.alpha}, beta={args.beta})")
                    for data, _ in test_loader:
                        data = data.to(device)
                        probs, _, _ = model(data)
                        all_probs.append(probs.cpu().numpy())
                    print("    Done!")

                all_probs = np.concatenate(all_probs, axis=0)
                top_2_probs = get_top_2_predictions(all_probs)
                test_filenames = test_loader.dataset.filenames
                test_filenames = [f"{filename}.mov" for filename in test_filenames]

                alpha_best = args.alpha
                beta_best = args.beta
                final_preds = probs2dict(top_2_probs, test_filenames, alpha_best, beta_best)
                final_preds = {"predictions": final_preds}

                # path_test_file_predictions = "data/{}_test_predictions.json".format(encoder)
                                              
                # path_test_file_predictions = f"data/{encoder}_{model_type}_fold{int(fold_id)}_test_predictions.json"
                path_test_file_predictions = f"{os.path.dirname(best_model_path)}/{os.path.splitext(os.path.basename(best_model_path))[0]}_test_predictions.json"
                print(f"Saving test predictions: \'{path_test_file_predictions}\'")
                with open(path_test_file_predictions, "w") as f:
                    json.dump(final_preds, f, indent=4)
                print("    Done!")

            else:
                acc_presence, acc_salience = train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha_best, beta_best, encoder)

            # print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")
            # print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")

            # # Save test results
            # test_summary_rows.append({
            #     "encoder": encoder,
            #     "model": model_type,
            #     "alpha": alpha_best,
            #     "beta": beta_best,
            #     "test_acc_presence": acc_presence,
            #     "test_acc_salience": acc_salience,
            # })

    # # Save test results
    # test_summary_df = pd.DataFrame(test_summary_rows)
    # test_summary_df.to_csv("test_summary.csv", index=False)
    # print("\nTest Summary:")
    # print(test_summary_df)

    # # Encoder/model-averaged test results
    # print("\nAveraged Test Results:")
    # print(test_summary_df.groupby(["encoder", "model"])[["test_acc_presence", "test_acc_salience"]].mean())




def main(do_val=True, do_test=False, args=None):
    # vision_encoders = ["imagebind", "videomae", "videoswintransformer", "openface", "clip"]
    # vision_encoders = ["imagebind"]
    # vision_encoders = ["imagebind11statistics"]
    # vision_encoders = ["videomae_hubert"]
    # vision_encoders = ["bfm"]
    # vision_encoders = ["bfm_transfer_single_exp"]
    vision_encoders = ["bfm_transfer_all_exp"]
    
    # audio_encoders = ["wavlm", "hubert"]
    audio_encoders = []

    # encoder_fusions = ["imagebind_wavlm", "imagebind_hubert", "videomae_wavlm", "videomae_hubert"]
    encoder_fusions = []

    encoders = vision_encoders + audio_encoders + encoder_fusions

    # model_types = ["Linear", "MLP_256", "MLP_512"]
    model_types = ["MLP_512"]


    if do_val:
        print(f"Loading train protocol \'{train_metadata_path}\'")
        train_df = pd.read_csv(train_metadata_path)
        train_labels = create_labels(train_df.to_dict(orient="records"))

        run_validation(train_df, train_labels, encoders, model_types, args)

    if do_test:
        print(f"Loading train protocol \'{train_metadata_path}\'")
        train_df = pd.read_csv(train_metadata_path)
        train_labels = create_labels(train_df.to_dict(orient="records"))

        print(f"Loading test protocol \'{train_metadata_path}\'")
        test_df = pd.read_csv(test_metadata_path)
        # test_labels = create_labels(test_df.to_dict(orient="records"))
        # run_test(train_df, train_labels, test_df, test_labels, encoders, model_types, use_best_model_from_val=False)
        get_test_preds(train_df, train_labels, test_df, encoders, model_types, use_best_model_from_val=True, args=args)


if __name__ == "__main__":
    args = parse_args()
    # main(do_val=True, do_test=False, args=args)
    
    if not args.test_model:
        do_val  = True
        do_test = False
    else:
        assert os.path.isfile(args.test_model), f"Error, no such file \'{args.test_model}\'"
        assert args.alpha > 0.0, f"Error args.alpha ({args.alpha}) <= 0.0"
        assert args.beta  > 0.0, f"Error args.beta  ({args.alpha}) <= 0.0"
        do_val  = False
        do_test = True

    main(do_val, do_test, args=args)
