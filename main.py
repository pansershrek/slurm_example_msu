from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
import traceback
import time
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def main():
    try:
        dataset = load_dataset("yelp_review_full", split={"train": "train[:100]", "test": "test[:100]"} )
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
        model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
        training_args = TrainingArguments(
            eval_strategy="epoch",
            push_to_hub=True,
            save_strategy="epoch",
            output_dir="result",
            num_train_epochs = 1,
            logging_strategy = "no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
        )
        trainer.train()
    except Exception as e:
        print(traceback.format_exc())
        print(e)

    time.sleep(10)
    print("FINISHED", flush=True)
    time.sleep(10)

if __name__ == "__main__":
    main()


