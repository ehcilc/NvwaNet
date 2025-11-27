#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Import our new Nvwa modules
from nvwa.module import NvwaNetSystem
from nvwa.datamodule import NvwaNetDataModule

# Set environment variables for better compatibility
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"

# --- Configuration ---
SEED_VAL = 42
pl.seed_everything(SEED_VAL)

# Paths
ROOT_DIR = "/media/ggj/WHY3/Project/AI/CodeTest/geneformer_pipeline/Version_nvwanet_gemini"
DATASET_PATH = "/media/ggj/WHY3/Project/AI/Model_survey/GeneformerV1/HCT116_NTC_5k/HCT116_NTC_5k.dataset"
TOKEN_DICT_PATH = "/media/ggj/WHY3/Project/AI/Model_survey/GeneformerV1/HCT116_coding_miRNA_token_dictionary.pkl"
RUN_NAME = "HCT116_NTC_5k"

OUTPUT_DIR = f"{ROOT_DIR}/models/{RUN_NAME}/"
LOGGING_DIR = f"{ROOT_DIR}/runs/{RUN_NAME}/"

# Model Hyperparameters
MAX_INPUT_SIZE = 4096
NUM_LAYERS = 12
NUM_ATTN_HEADS = 12
NUM_EMBED_DIM = 768
INTERMED_SIZE = 3072
ACTIV_FN = "relu"
INITIALIZER_RANGE = 0.02
LAYER_NORM_EPS = 1e-12
ATTENTION_PROBS_DROPOUT_PROB = 0.1
HIDDEN_DROPOUT_PROB = 0.1

# Training Hyperparameters
NUM_GPUS = 1
BATCH_SIZE = 8
MAX_LR = 1e-3
WARMUP_STEPS = 10_000
EPOCHS = 100
WEIGHT_DECAY = 0.001
# Estimated total steps (Used for LR scheduler)
# Adjust this based on your dataset size: num_examples / (batch_size * num_gpus) * epochs
ESTIMATED_NUM_EXAMPLES = 5000 
TOTAL_STEPS = int((ESTIMATED_NUM_EXAMPLES / (BATCH_SIZE * NUM_GPUS)) * EPOCHS)

def main():
    # 1. Load Token Dictionary
    with open(TOKEN_DICT_PATH, "rb") as fp:
        token_dictionary = pickle.load(fp)
    
    pad_token_id = token_dictionary.get("<pad>")
    vocab_size = len(token_dictionary)

    # 2. Config Dict for BERT
    model_config = {
        "hidden_size": NUM_EMBED_DIM,
        "num_hidden_layers": NUM_LAYERS,
        "initializer_range": INITIALIZER_RANGE,
        "layer_norm_eps": LAYER_NORM_EPS,
        "attention_probs_dropout_prob": ATTENTION_PROBS_DROPOUT_PROB,
        "hidden_dropout_prob": HIDDEN_DROPOUT_PROB,
        "intermediate_size": INTERMED_SIZE,
        "hidden_act": ACTIV_FN,
        "max_position_embeddings": MAX_INPUT_SIZE,
        "model_type": "bert",
        "num_attention_heads": NUM_ATTN_HEADS,
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
    }

    # 3. Initialize DataModule
    dm = NvwaNetDataModule(
        dataset_path=DATASET_PATH,
        token_dictionary=token_dictionary,
        batch_size=BATCH_SIZE,
        num_workers=8, # Adjust based on CPU cores
        seed=SEED_VAL
    )

    # 4. Initialize System (LightningModule)
    model = NvwaNetSystem(
        config_dict=model_config,
        learning_rate=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS
    )

    # 5. Callbacks & Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, "checkpoints"),
        filename="nvwanet-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger = TensorBoardLogger(save_dir=LOGGING_DIR, name=RUN_NAME)

    # 6. Initialize Trainer with DDP Strategy
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=NUM_GPUS,
        strategy="ddp", # Distributed Data Parallel
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed", # Enable AMP for speed and memory efficiency
        log_every_n_steps=50,
        enable_progress_bar=True
    )

    # 7. Start Training
    print("Starting NvwaNet Training...")
    trainer.fit(model, datamodule=dm)

    # 8. Save HuggingFace Format Model for Downstream Tasks
    # Only rank 0 needs to save the final HF model structure
    if trainer.global_rank == 0:
        print("Training finished. Saving HuggingFace model format...")
        hf_save_path = os.path.join(OUTPUT_DIR, "final_hf_model")
        model.save_hf_model(hf_save_path)
        print(f"HuggingFace model saved to: {hf_save_path}")

if __name__ == "__main__":
    main()