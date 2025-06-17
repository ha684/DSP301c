#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3-4B with UnsLoTH + LoRA, then **test its
function-calling behaviour** in four modes:

1.  Irrelevant function · no thinking
2.  Irrelevant function · thinking
3.  Relevant   function · no thinking
4.  Relevant   function · thinking

Run:
    python train_qwen_finetune.py --max_steps 30
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import argparse, re, json, random, os, torch, math
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

# --------------------------------------------------------------------------- #
# Pre-processing helpers
# --------------------------------------------------------------------------- #
def generate_conversation(examples):
    """Convert function-calling raw text → list[{role,content}]"""
    system_prompts, chats = examples["system"], examples["chat"]
    conversations = []
    for sys_raw, chat in zip(system_prompts, chats):
        convo = []
        sys_clean = re.sub(r'^\s*SYSTEM:\s*', '', sys_raw or '').strip()
        if sys_clean:
            convo.append({"role": "system", "content": sys_clean})

        for turn in re.split(r'USER:\s*', chat)[1:]:
            user_part, *assistant_parts = re.split(r'ASSISTANT:\s*', turn)

            user_msg = re.sub(r'<\|endoftext\|>', '', user_part).strip()
            if user_msg:
                convo.append({"role": "user",
                              "content": re.sub(r'\s+', ' ', user_msg)})

            for a in assistant_parts:
                a_clean = re.sub(r'<\|endoftext\|>', '', a).strip()
                # Split out any FUNCTION RESPONSE blocks
                func = re.search(r'FUNCTION RESPONSE:\s*(.*?)(?=ASSISTANT:|USER:|$)',
                                 a_clean, re.DOTALL)
                if func:
                    before = a_clean[:func.start()].strip()
                    tool   = func.group(1).strip()
                    after  = a_clean[func.end():].strip()
                    if before: convo.append({"role": "assistant", "content": before})
                    if tool:   convo.append({"role": "user",
                                             "content": f"tool response: {tool}"})
                    if after:  convo.append({"role": "assistant", "content": after})
                else:
                    if a_clean:
                        convo.append({"role": "assistant", "content": a_clean})
        conversations.append(convo)
    return {"conversations": conversations}


def reasoning_formatting(examples):
    """OpenMathReasoning-mini: problem/solution → 2-turn chat"""
    problems, solutions = examples["problem"], examples["generated_solution"]
    conversations = [[{"role": "user", "content": p},
                      {"role": "assistant", "content": s}]
                     for p, s in zip(problems, solutions)]
    return {"conversations": conversations}


def simple_exact_match(pred, label):
    """Exact-string matching metric stub."""
    return int(pred.strip() == label.strip())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--function_call_pct", type=float, default=0.25)
    p.add_argument("--eval_pct", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load datasets --------------------------------------------------------
    reasoning_dataset   = load_dataset("unsloth/OpenMathReasoning-mini",
                                       split="cot")
    functioncall_ds_raw = load_dataset("glaiveai/glaive-function-calling-v2",
                                       split="train")

    # 2) Load base model & tokenizer -----------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.model_name,
        max_seq_length  = args.max_seq_length,
        load_in_4bit    = True,
        load_in_8bit    = False,
        full_finetuning = False,
    )

    # 3) Attach LoRA adapters -------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r                       = 32,
        target_modules          = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
        lora_alpha              = 32,
        lora_dropout            = 0,
        bias                    = "none",
        use_gradient_checkpointing = "unsloth",
        random_state            = 3407,
        use_rslora              = False,
        loftq_config            = None,
    )

    # 4) Pre-process datasets -------------------------------------------------
    processed_functioncall = functioncall_ds_raw.map(
        generate_conversation, batched=True)["conversations"]
    processed_reasoning = reasoning_dataset.map(
        reasoning_formatting, batched=True)["conversations"]

    reasoning_strs  = tokenizer.apply_chat_template(processed_reasoning,
                                                    tokenize=False)
    function_strs   = tokenizer.apply_chat_template(processed_functioncall,
                                                    tokenize=False)

    fc_subset = pd.Series(function_strs).sample(
        int(len(reasoning_strs) *
            (args.function_call_pct / (1 - args.function_call_pct))),
        random_state=2407
    )
    data = pd.concat([pd.Series(reasoning_strs), fc_subset]).reset_index(drop=True)
    data.name = "text"

    # 5) Train / eval split ---------------------------------------------------
    df = pd.DataFrame(data).sample(frac=1, random_state=3407)  # shuffle
    n_eval   = int(len(df) * args.eval_pct)
    eval_df  = df.iloc[:n_eval]
    train_df = df.iloc[n_eval:]

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "eval" : Dataset.from_pandas(eval_df,  preserve_index=False),
    })

    # 6) Training -------------------------------------------------------------
    trainer = SFTTrainer(
        model          = model,
        tokenizer      = tokenizer,
        train_dataset  = dataset_dict["train"],
        eval_dataset   = dataset_dict["eval"],
        args = SFTConfig(
            output_dir                  = args.output_dir,
            dataset_text_field          = "text",
            per_device_train_batch_size = args.per_device_batch_size,
            gradient_accumulation_steps = args.gradient_accum,
            warmup_steps                = 5,
            # max_steps                  = args.max_steps,  # optional override
            learning_rate               = args.lr,
            logging_steps               = 1,
            optim                       = "adamw_8bit",
            weight_decay                = 0.01,
            lr_scheduler_type           = "cosine",
            seed                        = 3407,
            report_to                   = "none",
            fp16                        = True,
            dataloader_num_workers      = 4,
            eval_on_start               = True,
            # num_train_epochs            = 1,
            save_strategy               = "steps",
            save_steps                  = 100,
            save_total_limit            = 2,
            max_steps                   = 100,
        ),
    )

    trainer_stats = trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))

    # ----------------------------------------------------------------------- #
    # 7) Quick sanity-check: function-calling                                 #
    # ----------------------------------------------------------------------- #
    def run_generation(title: str, conversation: list[dict], thinking: bool):
        """Format chat, generate, and print a separator."""
        print(f"\n====== {title} | thinking={thinking} ======")
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize              = False,
            add_generation_prompt = True,
            enable_thinking       = thinking,
        )
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to(model.device),
            max_new_tokens = 256 if not thinking else 1024,
            temperature    = 0.6 if thinking else 0.7,
            top_p          = 0.95 if thinking else 0.8,
            top_k          = 20,
            streamer       = TextStreamer(tokenizer, skip_prompt=True),
        )

    # Shared function schema --------------------------------------------------
    fn_schema = (
        "You are a helpful assistant with access to the following function. "
        "Use it **only** when it helps the user.\n"
        "{\n"
        '  "name": "get_exchange_rate",\n'
        '  "description": "Get the exchange rate between two currencies",\n'
        '  "parameters": {\n'
        '    "type": "object",\n'
        '    "properties": {\n'
        '      "base_currency":   {"type": "string", '
        '"description": "Currency to convert from"},\n'
        '      "target_currency": {"type": "string", '
        '"description": "Currency to convert to"}\n'
        '    },\n'
        '    "required": ["base_currency", "target_currency"]\n'
        "  }\n"
        "}"
    )

    # Case A ─ function is **irrelevant**
    msgs_irrelevant = [
        {"role": "system", "content": fn_schema},
        {"role": "user",
         "content": ("Can you book a flight for me from New York to London "
                     "next Tuesday?")}
    ]
    run_generation("IRRELEVANT-FUNCTION", msgs_irrelevant, thinking=False)
    run_generation("IRRELEVANT-FUNCTION", msgs_irrelevant, thinking=True)

    # Case B ─ function is **relevant**
    msgs_relevant = [
        {"role": "system", "content": fn_schema},
        {"role": "user",
         "content": "What's the exchange rate from USD to EUR right now?"}
    ]
    run_generation("RELEVANT-FUNCTION", msgs_relevant, thinking=False)
    run_generation("RELEVANT-FUNCTION", msgs_relevant, thinking=True)

    # ----------------------------------------------------------------------- #
    # 8) Plot training / eval loss ------------------------------------------- #
    history = pd.DataFrame(trainer.state.log_history)  # step,loss,eval_loss,...
    plt.figure()
    plt.plot(history["step"], history["loss"], label="train_loss")
    if "eval_loss" in history:
        plt.plot(history["step"], history["eval_loss"], label="eval_loss")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Training & Eval loss")
    plt.legend()
    png_path = os.path.join(args.output_dir, "training_metrics.png")
    plt.savefig(png_path, dpi=150)
    print(f"\nSaved plot to {png_path}")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
