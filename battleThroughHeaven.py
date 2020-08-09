from os import system
system('watch -n 20 -d nvidia-smi')


from transformers import BertTokenizer
tokenizer = BertTokenizer('vocab_small.txt')

from transformers import GPT2Config
from transformers import GPT2LMHeadModel
config = GPT2Config(vocab_size=52000)
device = "cuda"
model = GPT2LMHeadModel(config=config)
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
    num_train_epochs=10,
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

system('zip -q -r doupoGPT2.zip doupoGPT2/')