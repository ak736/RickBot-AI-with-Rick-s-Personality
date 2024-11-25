import logging
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Main training function"""
    
    # Set up TensorBoard
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Calculate batch size
    args.train_batch_size = args.per_gpu_train_batch_size

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Create data loader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate,
        drop_last=True
    )

    # Set up training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare model for training
    model = model.module if hasattr(model, "module") else model
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=t_total
    )

    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Training batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            
            # Move batch to device (CPU/MPS for M2 Mac)
            inputs = batch.to(args.device)
            labels = batch.to(args.device)
            
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", tr_loss / args.logging_steps, global_step)
                    tr_loss = 0.0

                # Saving model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
                
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step