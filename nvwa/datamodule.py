import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from .collator import NvwaPreCollator

class NvwaNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        token_dictionary: dict,
        batch_size: int = 8,
        num_workers: int = 4,
        mlm_probability: float = 0.15,
        val_split_ratio: float = 0.1,  # Option to split train dataset if no explicit val set
        seed: int = 42
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.token_dictionary = token_dictionary
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mlm_probability = mlm_probability
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage: str = None):
        # Load dataset from disk
        full_dataset = load_from_disk(self.dataset_path)
        
        # --- 数据集分割逻辑 ---
        if "train" not in full_dataset and "validation" not in full_dataset:
            # Check if it's a DatasetDict or just a Dataset
            if hasattr(full_dataset, "train_test_split"):
                split_ds = full_dataset.train_test_split(test_size=self.val_split_ratio, seed=self.seed)
                self.train_dataset = split_ds["train"]
                self.val_dataset = split_ds["test"]
            else:
                # Assume it's already just one split, treat as train
                self.train_dataset = full_dataset
                # Warning: No validation set
        else:
            self.train_dataset = full_dataset.get("train", full_dataset)
            self.val_dataset = full_dataset.get("validation", None)

        # --- 关键修复：过滤掉非 Tensor 友好的列（如 'Anno' 等元数据） ---
        # 我们只保留模型训练真正需要的列
        target_columns = ["input_ids", "length", "attention_mask", "token_type_ids"]
        
        def filter_columns(dataset):
            if dataset is None: 
                return None
            # 找出该数据集中实际存在的、且是我们需要的列
            cols_to_keep = [c for c in target_columns if c in dataset.column_names]
            return dataset.select_columns(cols_to_keep)

        self.train_dataset = filter_columns(self.train_dataset)
        self.val_dataset = filter_columns(self.val_dataset)
        # -----------------------------------------------------------

        # Initialize Collator
        # 使用修正后的 NvwaPreCollator (确保你已经更新了 collator.py)
        self.precollator = NvwaPreCollator(token_dictionary=self.token_dictionary)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.precollator, 
            mlm=True, 
            mlm_probability=self.mlm_probability
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.data_collator,
                pin_memory=True
            )
        return None