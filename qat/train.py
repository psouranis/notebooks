import hydra
import os
from omegaconf import DictConfig

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    ToTensor,
)

from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    Trainer, 
    TrainingArguments,
    DefaultDataCollator
)
from utils import transforms, create_label_to_ids, compute_metrics
from functools import partial

os.environ["HYDRA_FULL_ERROR"] = "1" 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    data = load_dataset(cfg.dataset_path)
    labels = data["train"].features["label"].names
    
    label2id, id2label = create_label_to_ids(labels)
    
    base_preprocessor = AutoImageProcessor.from_pretrained(cfg.model_name)
    
    normalize = Normalize(mean=base_preprocessor.image_mean, std=base_preprocessor.image_std)
    size = (
        base_preprocessor.size["shortest_edge"]
        if "shortest_edge" in base_preprocessor.size
        else (base_preprocessor.size["height"], base_preprocessor.size["width"])
    )
    
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    transform_func = partial(transforms, _transforms=_transforms)
    
    data = data.with_transform(transform_func)
    data_collator = DefaultDataCollator()
    
    model = AutoModelForImageClassification.from_pretrained(
                cfg.model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id
    )
    
    training_args = TrainingArguments(
                **cfg.trainer,
                remove_unused_columns=False,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=base_preprocessor,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # use torch.save instead of trainer.save_model

if __name__ == "__main__":
    main()    

