from transformers import BertTokenizer
tokenizer = BertTokenizer('vocab_small.txt')

from transformers import GPT2LMHeadModel
device = "cuda"
model = GPT2LMHeadModel.from_pretrained('./doupoGPT2')
model.to(device)
model.train()
print(model.num_parameters())

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./doupo.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./doupoGPT2",
    overwrite_output_dir=True,
    logging_steps=50,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)
trainer.train()
trainer.save_model("./doupoGPT2")
