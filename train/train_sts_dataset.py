from torch.utils.data import DataLoader, Dataset
import logging
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TrainDataset(Dataset):
    def __init__(self, data, args):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx][0],  # anchor
            self.data[idx][1],  # positive
            self.data[idx][2],  # negative
        )


def init_dataloader(args):
    data_set_dict = {
        "wiki_en": {
            "triplet": {
                "train": "data/wiki_en_sentence/preprocessed/train_data_triplets.json",
                "dev": "data/wiki_en_sentence/preprocessed/val_data_pair.json",
                "train_class": TrainDataset,
                "dev_class": TrainDataset,
            },
            "pair": {
                "train": "data/wiki_en_sentence/preprocessed/train_data_pairs.json",
                "dev": "data/wiki_en_sentence/preprocessed/val_data_pair.json",
                "train_class": TrainDataset,
                "dev_class": TrainDataset,
            },
        },
    }

    assert args.data_set in data_set_dict, f"Data set must be in {data_set_dict.keys()}"

    train_data = read_json(data_set_dict[args.data_set][args.mode]["train"])
    val_data = read_json(data_set_dict[args.data_set][args.mode]["dev"])

    train_data = data_set_dict[args.data_set][args.mode]["train_class"](
        train_data, args
    )
    val_data = data_set_dict[args.data_set][args.mode]["dev_class"](val_data, args)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        # pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logging.info(f"Train data size: {len(train_data)}")
    logging.info(f"Validation data size: {len(val_data)}")
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    class Args:
        data_set = "wiki_en"
        mode = "triplet"
        embedding_model = "BAAI/bge-base-en-v1.5"
        batch_size = 32

    args = Args()
    train_loader, val_loader = init_dataloader(args)
