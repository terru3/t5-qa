import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForQuestionAnswering, get_scheduler

from constants import *
from preprocessing import train_dataloader, val_dataloader
from utils import set_seed, device

"""# Training setup"""

# TODO: allow LoRA=true/false, model path, choice of optimizer and LR scheduler(s), etc.
def prep_train():

    """
    Prepares model, optimizer, LR scheduler for training

    """
    model = AutoModelForQuestionAnswering.from_pretrained("t5-small")
    model.to(device)

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        inference_mode=False,
        r=8,  # matrix rank
        lora_alpha=32,  # scaling factor
        lora_dropout=0.1,
    )

    # Wrap model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

    # Initialize scheduler
    num_training_steps = EPOCHS * len(train_dataloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader)
        * NUM_EPOCH_WARMUP,  # half an epoch for warmup
        num_training_steps=num_training_steps,
    )

    return model, optimizer, scheduler


"""# Training"""


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    """
    Trains the model with the specified optimizer and LR scheduler.
    Returns the trained model as well as lists of train and validation losses.
    """
    model.train()
    train_losses = []
    val_losses = []

    # calculate number of times to compute train/val loss
    COMPUTE_EVERY = round(
        int(len(train_dataloader) / COMPUTE_PER_EPOCH), -2
    )  # nearest hundred
    print(
        f"There are {len(train_dataloader)} batches in an epoch; train and val losses are computed every",
        f"{COMPUTE_EVERY} steps.",
    )

    for epoch in range(EPOCHS):
        print(
            f"Epoch {epoch}, Learning rate: {scheduler.optimizer.param_groups[0]['lr']:.5f}"
        )
        epoch_loss = []
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            # outputs has keys 'loss', 'start_logits', 'end_logits', 'encoder_last_hidden_state'
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            stop = time.time()

            if step % COMPUTE_EVERY == 0:
                train_losses.append(loss.item())
                # Calculate validation loss on a single batch only for compute reasons
                val_batch = next(iter(val_dataloader))
                with torch.no_grad():
                    model.eval()
                    output = model(
                        val_batch["input_ids"].to(device),
                        attention_mask=val_batch["attention_mask"].to(device),
                        start_positions=val_batch["start_positions"].to(device),
                        end_positions=val_batch["end_positions"].to(device),
                    )
                    val_loss = output.loss
                    val_losses.append(val_loss.item())
                model.train()

                # Print statistics
                print(
                    f"Epoch: {epoch+1}/{EPOCHS} | Step: {step}/{len(train_dataloader)} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f} |",
                    f"LR: {scheduler.optimizer.param_groups[0]['lr']:.5f} | Time of Last Batch: {stop-start:.2f} \n",
                )

        avg_train_loss = total_loss / len(train_dataloader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

    model.save_pretrained(f"{PATH}/t5-squad-tuned")
    return model, train_losses, val_losses


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(12, 8))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.xticks(
        ticks=plt.xticks()[0][1:], labels=400 * np.array(plt.xticks()[0][1:], dtype=int)
    )  # steps * 400
    plt.legend()
    plt.show()
    plt.savefig(f"{PATH}/figures/loss.png")
