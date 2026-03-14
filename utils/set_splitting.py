import numpy as np
from sklearn.preprocessing import StandardScaler
from datasets.d2_dataset import D2Dataset
from datasets.subsample_dataset import SubsampledVideoDataset
<<<<<<< HEAD
from config import INDEX_TO_LABEL, NEUTRAL_INDEX
=======

>>>>>>> origin/biesseck

def get_validation_split(df, labels, fold_id):
    """Split filenames and labels into train/val sets based on fold."""
    train_mask = df["fold"] != fold_id
    val_mask = ~train_mask

    train_files = df.loc[train_mask, "filename"].tolist()
    val_files = df.loc[val_mask, "filename"].tolist()

    train_labels = labels[train_mask.to_numpy()]
    val_labels = labels[val_mask.to_numpy()]

    return (train_files, train_labels), (val_files, val_labels)


def prepare_split_2d(train_files, train_labels, val_files, val_labels, filepath):
<<<<<<< HEAD
    data = np.load(filepath)
    X = data["X"]
    filenames = data["filenames"]
=======
    print(f"Loading train/val data '{filepath}'")
    data = np.load(filepath)
    X = data["X"]
    filenames = data["filenames"]
    print(f"    Done    (X.shape: {X.shape},    filenames.shape: {filenames.shape})")
>>>>>>> origin/biesseck

    # Map filenames to indices
    name_to_idx = {name: i for i, name in enumerate(filenames)}
    train_idx = [name_to_idx[f] for f in train_files]
    val_idx = [name_to_idx[f] for f in val_files]

    # Subset and scale
<<<<<<< HEAD
    X_train = X[train_idx]
    X_val = X[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
=======
    print(f"Splitting train data")
    X_train = X[train_idx]
    print(f"    X_train.shape: {X_train.shape}")
    print(f"Splitting val data")
    X_val = X[val_idx]
    print(f"    X_val.shape: {X_val.shape}")

    if len(X_train.shape) <= 2:
        print(f"Normalizing train/val data")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
>>>>>>> origin/biesseck

    train_dataset = D2Dataset(X=X_train, labels=train_labels, filenames=train_files)
    val_dataset = D2Dataset(X=X_val, labels=val_labels, filenames=val_files)

    return train_dataset, val_dataset

<<<<<<< HEAD
def prepare_test_2d(test_files, filepath, scaler):
    """
    Prepares the test dataset using features and a pre-fitted scaler.
    """
    data = np.load(filepath)
    X = data["X"]
    filenames = data["filenames"]

    # Map filenames to indices
    name_to_idx = {name: i for i, name in enumerate(filenames)}
    
    # We use a list comprehension with a check to ensure filename exists in .npy
    test_idx = [name_to_idx[f] for f in test_files if f in name_to_idx]

    X_test = X[test_idx]
    X_test = scaler.transform(X_test)

    # Create dummy labels (zeros) since test set has no ground truth
    # Shape: (Number of test samples, Number of classes)
    num_classes = len(INDEX_TO_LABEL)
    test_labels = np.zeros((len(test_idx), num_classes))

    test_dataset = D2Dataset(X=X_test, labels=test_labels, filenames=test_files)

    return test_dataset


def prepare_train_2d(train_files, train_labels, filepath):
    data = np.load(filepath)
    X = data["X"]
    filenames = data["filenames"]

    # Map filenames to indices
    name_to_idx = {name: i for i, name in enumerate(filenames)}
    train_idx = [name_to_idx[f] for f in train_files]

    # Subset and scale
    X_train = X[train_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    train_dataset = D2Dataset(X=X_train, labels=train_labels, filenames=train_files)

    return train_dataset, scaler 
=======
'''
# FOR FRAME CLASSIFICATION
def prepare_split_2d(train_files, train_labels, val_files, val_labels, filepath):
    print(f"Loading train/val data '{filepath}'")
    data = np.load(filepath)
    X = data["X"]
    filenames = data["filenames"]
    print(f"    Done    (X.shape: {X.shape},    filenames.shape: {filenames.shape})")

    train_frames = np.concatenate([filenames[filenames==f] for f in train_files], axis=0)
    val_frames   = np.concatenate([filenames[filenames==f] for f in val_files], axis=0)
    
    # Map filenames to indices
    # name_to_idx = {name: i for i, name in enumerate(filenames)}
    # train_idx = [name_to_idx[f] for f in train_files]
    # val_idx = [name_to_idx[f] for f in val_files]
    train_idx = np.concatenate([np.where(filenames==f)[0] for f in train_files], axis=0)
    val_idx   = np.concatenate([np.where(filenames==f)[0] for f in val_files], axis=0)

    train_labels_frames = np.vstack([[train_labels[i]] * int(np.sum(filenames==file)) for i, file in enumerate(train_files)])
    val_labels_frames   = np.vstack([[val_labels[i]]   * int(np.sum(filenames==file)) for i, file in enumerate(val_files)])

    # Subset and scale
    print(f"Splitting train data")
    X_train = X[train_idx]
    print(f"    X_train.shape: {X_train.shape}")
    print(f"Splitting val data")
    X_val = X[val_idx]
    print(f"    X_val.shape: {X_val.shape}")

    if len(X_train.shape) <= 2:
        print(f"Normalizing train/val data")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    train_dataset = D2Dataset(X=X_train, labels=train_labels_frames, filenames=train_frames)
    val_dataset = D2Dataset(X=X_val, labels=val_labels_frames, filenames=val_frames)

    return train_dataset, val_dataset
'''
>>>>>>> origin/biesseck

def prepare_split_subsampled(df, labels, fold_id, data_dir):
    (train_files, train_labels), (val_files, val_labels) = get_validation_split(df, labels, fold_id)

    train_dataset = SubsampledVideoDataset(filenames=train_files, labels=train_labels, data_dir=data_dir)
    val_dataset = SubsampledVideoDataset(filenames=val_files, labels=val_labels, data_dir=data_dir)

    # Scale
    scaler = StandardScaler()
    train_dataset.features = scaler.fit_transform(train_dataset.features)
    val_dataset.features = scaler.transform(val_dataset.features)

    return train_dataset, val_dataset