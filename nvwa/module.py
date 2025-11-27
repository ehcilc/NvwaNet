import pytorch_lightning as pl
import torch
from transformers import BertConfig, BertForMaskedLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class NvwaNetSystem(pl.LightningModule):
    def __init__(
        self,
        config_dict: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.001,
        warmup_steps: int = 10000,
        total_steps: int = 100000,
    ):
        """
        NvwaNet Lightning Module.
        Args:
            config_dict: Dictionary containing BertConfig parameters.
            learning_rate: Maximum learning rate.
            weight_decay: Weight decay for optimizer.
            warmup_steps: Number of warmup steps for scheduler.
            total_steps: Total number of training steps (for scheduler).
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.bert_config = BertConfig(**config_dict)
        self.model = BertForMaskedLM(self.bert_config)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def training_step(self, batch, batch_idx):
        # Filter out keys that BertForMaskedLM doesn't accept (e.g., 'length')
        model_inputs = {
            k: v for k, v in batch.items() 
            if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
        }
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Filter out keys that BertForMaskedLM doesn't accept (e.g., 'length')
        model_inputs = {
            k: v for k, v in batch.items() 
            if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
        }
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Separate parameters for weight decay handling
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        # Linear warmup schedule
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps
        )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [scheduler_config]
    
    def on_save_checkpoint(self, checkpoint):
        """
        Custom hook to ensure HF model can be loaded by legacy scripts.
        """
        # We can also save the config here if needed, but the PL checkpoint contains hparams.
        pass

    def save_hf_model(self, output_dir):
        """
        Helper to save the internal HF model for downstream tasks (perturber, etc.)
        """
        self.model.save_pretrained(output_dir)