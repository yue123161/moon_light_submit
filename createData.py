import os
import random
import json
import gzip
from torch_geometric.data import Data
import torch
import random
import csv
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from utils import set_seed
import json
# Set seeds and parameters
set_seed()
torch.set_num_threads(1)
data_folder = "dataJson"
os.makedirs(data_folder, exist_ok=True)


def serialize_data(data):
    return {
        "edge_index": data.edge_index.tolist(),
        "edge_attr": data.edge_attr.tolist(),
        "num_nodes": data.num_nodes,
        "y": data.y.tolist() if hasattr(data, 'y') and data.y is not None else None
    }

def save_dataset_json(dataset, directory, filename):
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, f"{filename}.json.gz")
    
    # Serialize the dataset
    serialized_data = [serialize_data(data) for data in dataset]
    
    # Write compressed JSON
    with gzip.open(save_path, "wt", encoding="utf-8") as f:
        json.dump(serialized_data, f)
    
    print(f"Saved dataset with {len(dataset)} graphs to {save_path}")




# Function to introduce symmetric label noise
def add_label_noise(labels, noise_ratio, new_number_classes):
    num_labels = labels.size(0)
    num_noisy = int(noise_ratio * num_labels)
    noisy_indices = random.sample(range(num_labels), num_noisy)
    noisy_labels = labels.clone()

    for idx in noisy_indices:
        current_label = noisy_labels[idx].item()
        new_label = random.choice([l for l in range(new_number_classes) if l != current_label])
        noisy_labels[idx] = new_label

    return noisy_labels

# Function to introduce asymmetric label noise
def add_asymmetric_label_noise_exact(labels, noise_ratio, mapping):
    noisy_labels = labels.clone()
    for source_class, target_class in mapping.items():
        source_indices = [i for i, label in enumerate(labels) if label.item() == source_class]
        num_to_flip = int(noise_ratio * len(source_indices))
        flipped_indices = random.sample(source_indices, num_to_flip)
        for idx in flipped_indices:
            noisy_labels[idx] = target_class

    return noisy_labels

# Split class data for custom train/test splits
def split_class_data(data, class_label, split_ratios):
    class_data = [d for d in data if d.y.item() == class_label]
    random.shuffle(class_data)
    split_points = [int(r * len(class_data)) for r in split_ratios]
    return (
        class_data[:split_points[0]],  # First split
        class_data[split_points[0]:],  # Remaining
    )

# Function to save individual graphs to directories
# def save_graphs_to_dir(graphs, directory):
#     os.makedirs(directory, exist_ok=True)
#     for i, graph in enumerate(graphs):
#         torch.save(graph, os.path.join(directory, f"graph_{i}.pt"))
#     print(f"Saved {len(graphs)} graphs to {directory}")

def save_dataset(dataset, directory, filename, metadata=None):
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)
    torch.save(dataset, save_path)
    print(f"Saved dataset with {len(dataset)} graphs to {save_path}")

    # Save metadata if provided
    if metadata:
        metadata_path = os.path.join(directory, f"{filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to {metadata_path}")

def save_dataset_in_chunks(dataset, directory, filename_base, chunk_size=1000):
    os.makedirs(directory, exist_ok=True)
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size  # Calculate number of chunks
    for i in range(num_chunks):
        chunk = dataset[i * chunk_size:(i + 1) * chunk_size]
        chunk_filename = f"{filename_base}_chunk_{i}.pt"
        torch.save(chunk, os.path.join(directory, chunk_filename))
        print(f"Saved chunk {i + 1}/{num_chunks} to {chunk_filename}")


# Conditions for noise addition
conditions = [
    # {"seed": 42, "sym": True, "noise": 0.2}
    # {"seed": 60, "sym": False, "noise": 0.2, "mapping": {2: 4, 4: 2}},
    # {"seed": 70, "sym": True, "noise": 0.4} ,
    {"seed": 231, "sym": False, "noise": 0.4, "mapping": {0: 4, 4: 0}}
]

for condition in conditions:
    # Load dataset and initialize
    dataset_name = "ogbg-ppa"
    dataset = PygGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(dataset_name)

    # Load and combine train/valid/test splits
    train_data = [dataset[i] for i in split_idx["train"]]
    valid_data = [dataset[i] for i in split_idx["valid"]]
    test_data = [dataset[i] for i in split_idx["test"]]
    train_data.extend(valid_data)
    num_classes = len(torch.cat([data.y for data in train_data]).unique())

    # Apply noise conditions
    set_seed(condition["seed"])
    noise = condition["noise"]
    sym = condition["sym"]
    selected_classes = sorted(random.sample(range(num_classes), 6))
    print(f"Selected classes: {selected_classes}")

    # Create new train, test, and blind test datasets
    new_train, new_test, blind_test = [], [], []
    for idx, cls_label in enumerate(selected_classes):
        cls_train, _ = split_class_data(train_data, cls_label, (0.4, 0.6))
        new_train.extend(cls_train)
        cls_test, blind_part = split_class_data(test_data, cls_label, (0.3, 0.3))
        new_test.extend(cls_test)
        blind_test.extend(blind_part)

    # Relabel data to the new class indices
    for idx, cls_label in enumerate(selected_classes):
        for d in new_train:
            if d.y.item() == cls_label:
                d.y[0] = idx
        for d in new_test:
            if d.y.item() == cls_label:
                d.y[0] = idx

    # Shuffle and add noise to training labels
    random.shuffle(new_train)
    new_number_classes = len(torch.cat([data.y for data in new_train]).unique())
    train_labels = torch.cat([data.y for data in new_train])

    if sym:
        train_labels_noisy_sym = add_label_noise(train_labels, noise, new_number_classes)
        for idx, data in enumerate(new_train):
            data.y[0] = train_labels_noisy_sym[idx]
    else:
        asymmetric_mapping = condition["mapping"]
        train_labels_noisy_asym = add_asymmetric_label_noise_exact(train_labels, noise, asymmetric_mapping)
        for idx, data in enumerate(new_train):
            data.y[0] = train_labels_noisy_asym[idx]

    # Shuffle and save test labels (CSV) after stripping them from graphs
    random.shuffle(new_test)
    save_labels = []
    for data in new_test:
        save_labels.append(data.y.item())
        del data.y  # Remove labels from test data

    save_labels_path = os.path.join(data_folder, f"{noise}_labels.csv" if sym else f"{noise}_asy_labels.csv")
    with open(save_labels_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for label in save_labels:
            writer.writerow([label])

    # Save datasets to directories
    # save_dataset(new_train, data_folder, f"{noise}_train.pt" if sym else f"{noise}_asy_train.pt",
    #              metadata={"noise_ratio": noise, "type": "symmetric" if sym else "asymmetric"})
    # save_dataset(new_test, data_folder, f"{noise}_test.pt" if sym else f"{noise}_asy_test.pt")
    # save_dataset(blind_test, data_folder, f"{noise}_blind_test.pt" if sym else f"{noise}_asy_blind_test.pt")
    
    # save_dataset_in_chunks(new_train, data_folder, f"{noise}_train" if sym else f"{noise}_asy_train")

    save_dataset_json(new_train, data_folder, f"{noise}_train" if sym else f"{noise}_asy_train")
    save_dataset_json(new_test, data_folder, f"{noise}_test" if sym else f"{noise}_asy_test")
    save_dataset_json(blind_test, data_folder, f"{noise}_blind_test" if sym else f"{noise}_asy_blind_test")
    



    print("Datasets processed and saved successfully!")
