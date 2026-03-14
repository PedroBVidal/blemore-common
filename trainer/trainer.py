<<<<<<< HEAD
from json import encoder
import json

import torch
import numpy as np
import os

from model.post_process import grid_search_thresholds, probs2dict, get_top_2_predictions
=======
import torch
import numpy as np
import os
import time

from model.post_process import grid_search_thresholds
>>>>>>> origin/biesseck
from utils.subsample_utils import aggregate_subsamples

class Trainer(object):

    def __init__(self, model, optimizer, data_loader, epochs, valid_data_loader=None, subsample_aggregation=False, save_dir="checkpoints"):
        self.model = model
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.data_loader = data_loader
        self.epochs = epochs
        self.valid_data_loader = valid_data_loader
        self.subsample_aggregation = subsample_aggregation

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_score = -np.inf  # Higher is better

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
<<<<<<< HEAD

=======
            if len(data.shape) == 2:
                data = torch.unsqueeze(data, 1)
            print(f'    train_epoch()    batch_idx: {batch_idx}/{len(self.data_loader)}    data.shape: {data.shape}    target.shape: {target.shape}', end='\r')
>>>>>>> origin/biesseck
            self.optimizer.zero_grad()
            probs, logits, loss = self.model(data, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
<<<<<<< HEAD
=======
        print()
>>>>>>> origin/biesseck

        avg_loss = total_loss / len(self.data_loader)
        return avg_loss

    def validate(self):
<<<<<<< HEAD
        print("\nValidating model...")
=======
        print("    Validating model...")
>>>>>>> origin/biesseck

        self.model.eval()
        total_loss = 0
        all_probs = []
        all_logits = []

        with torch.no_grad():
<<<<<<< HEAD
            for data, target in self.valid_data_loader:
                data, target = data.to(self.device), target.to(self.device)
=======
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                print('data.shape:', data.shape, '    target.shape:', target.shape)
                # if len(data.shape) == 2:
                #     data = torch.unsqueeze(data, 1)
                print(f'    validate()    batch_idx: {batch_idx}/{len(self.valid_data_loader)}    data.shape: {data.shape}    target.shape: {target.shape}', end='\r')
>>>>>>> origin/biesseck
                probs, logits, loss = self.model(data, target)
                total_loss += loss.item()
                all_probs.append(probs.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
<<<<<<< HEAD
=======
            print()
>>>>>>> origin/biesseck

        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        val_filenames = self.valid_data_loader.dataset.filenames

        if self.subsample_aggregation:
            val_filenames, all_probs = aggregate_subsamples(val_filenames, all_logits)
<<<<<<< HEAD
        

        ret = grid_search_thresholds(val_filenames, all_probs)
        alpha = ret['alpha']
        beta = ret['beta']

        top_2_probs = get_top_2_predictions(all_probs)
        final_preds = probs2dict(top_2_probs, val_filenames, alpha, beta)

        #with open("data/{}_val_predictions.json".format(encoder), "w") as f:
        #    json.dump(final_preds, f, indent=4)

        avg_loss = total_loss / len(self.valid_data_loader)
        ret["val_loss"] = avg_loss
        return ret,final_preds
=======

        print(f'    validate()    doing grid search of thresholds...')
        ret = grid_search_thresholds(val_filenames, all_probs)
        avg_loss = total_loss / len(self.valid_data_loader)
        ret["val_loss"] = avg_loss
        return ret
>>>>>>> origin/biesseck

    def train(self, writer=None, save_prefix="model"):
        best_epoch_stats = None
        best_model_path = None

<<<<<<< HEAD
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}")
=======
        total_time = 0.0
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"Epoch [{epoch + 1}/{self.epochs}]")
            train_loss = self.train_epoch()
            print(f"    Train Loss: {train_loss:.4f}")
>>>>>>> origin/biesseck

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)

            if self.valid_data_loader is not None:
<<<<<<< HEAD
                stats, final_preds = self.validate()

                val_score = 0.5 * stats['acc_presence'] + 0.5 * stats['acc_salience']  # scoring metric

                print(f"Epoch [{epoch + 1}/{self.epochs}], "
=======
                stats = self.validate()
                val_score = 0.5 * stats['acc_presence'] + 0.5 * stats['acc_salience']  # scoring metric

                print(f"    Epoch [{epoch + 1}/{self.epochs}], "
>>>>>>> origin/biesseck
                      f"Validation Loss: {stats['val_loss']:.4f}, "
                      f"Best Alpha: {stats['alpha']:.4f}, "
                      f"Best Beta: {stats['beta']:.4f}, "
                      f"Best Acc Presence: {stats['acc_presence']:.4f}, "
                      f"Best Acc Salience: {stats['acc_salience']:.4f}, "
                      f"Best possible Acc Presence : {stats['presence_only']:.4f}, "
                      f"Best possible Acc Salience : {stats['salience_only']:.4f}")

                if writer:
                    writer.add_scalar("Loss/val", stats['val_loss'], epoch)
                    writer.add_scalar("Accuracy/presence", stats['acc_presence'], epoch)
                    writer.add_scalar("Accuracy/salience", stats['acc_salience'], epoch)
                    writer.add_scalar("Alpha", stats['alpha'], epoch)
                    writer.add_scalar("Beta", stats['beta'], epoch)
                    writer.add_scalar("Best possible Accuracy/presence", stats['presence_only'], epoch)
                    writer.add_scalar("Best possible Accuracy/salience", stats['salience_only'], epoch)

                if val_score > self.best_val_score:
                    # Save best model
                    self.best_val_score = val_score
                    best_model_path = os.path.join(self.save_dir, f"{save_prefix}_best.pth")
<<<<<<< HEAD

                    with open("data/epoch_{}_{}_best_val_predictions.json".format(epoch,self.model.model_type), "w") as f:
                        json.dump(final_preds, f, indent=4)


=======
>>>>>>> origin/biesseck
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'input_dim': self.model.input_dim,
                        'output_dim': self.model.output_dim,
                        'model_type': self.model.model_type,
                    }, best_model_path)

                    best_epoch_stats = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": stats['val_loss'],
                        "best_alpha": stats['alpha'],
                        "best_beta": stats['beta'],
                        "best_acc_presence": stats['acc_presence'],
                        "best_acc_salience": stats['acc_salience'],
                    }
<<<<<<< HEAD
=======
            
            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time-epoch_start_time
            estimated_time = (self.epochs - epoch+1) * epoch_elapsed_time
            total_time += epoch_elapsed_time
            print()
            print(f"    Epoch elapsed time: {epoch_elapsed_time:.2f} sec    {epoch_elapsed_time/60:.2f} min    {epoch_elapsed_time/3600:.2f} hour")
            print(f"    Estimated time: {estimated_time:.2f} sec    {estimated_time/60:.2f} min    {estimated_time/3600:.2f} hour")
            print(f"    Total time: {total_time:.2f} sec    {total_time/60:.2f} min    {total_time/3600:.2f} hour")
            print('-------------')
>>>>>>> origin/biesseck

        return best_epoch_stats, best_model_path
