import numpy as np
import evaluate

def transforms(examples, _transforms):
    
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def create_label_to_ids(labels):
    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
        
    return label2id, id2label

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    f1 = evaluate.load("f1")

    return {
        "f1_macro": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1_micro": f1.compute(predictions=predictions, references=labels, average="micro")["f1"],
        "prec_macro": prec.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "prec_micro": prec.compute(predictions=predictions, references=labels, average="micro")["precision"],
        "accuracy": acc.compute(predictions=predictions, references=labels)["accuracy"]
    }