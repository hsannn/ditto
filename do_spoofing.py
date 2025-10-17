import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader
from typing import List
from pathlib import Path
from tqdm import tqdm
import os
import random
import argparse
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    return batch

def process_batch(process_id: int, time_stamp: int, prompts: List[str], args):
    """
    Worker process that generates text from the TEACHER model while applying
    the REVERSE watermark derived from the student model.
    """
    # Device
    device = f"cuda:{process_id}" if torch.cuda.is_available() else "cpu"
    
    # Create temporary output file
    temp_output = Path(args.output_file).parent / f"temp_{process_id}_{time_stamp}.jsonl"
    
    # --- MODIFIED: Load the TEACHER model ---
    # The reverse watermark will be applied to this model's outputs.
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # hsan: for debug
        low_cpu_mem_usage=True, 
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            device=model.device,
            max_new_tokens=args.max_new_tokens,
            min_length = 30 + args.max_new_tokens,
            # min_new_tokens=30,    
            # max_length=4096,      
            do_sample=True,       # False when greedy decoding
            temperature=0.7
        )
    
    # Load the reverse watermark derived from the student model
    watermark = AutoWatermark.load('Spoofing',
                                   args.reverse_watermark_config,
                                   transformers_config,)

    # Create dataset and dataloader for this process's portion
    dataset = TextDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    with open(temp_output, mode='w') as f:
        pbar = tqdm(total=len(dataloader), 
                    desc=f"Process {process_id} Generating",
                    position=process_id)

        for batch_prompts in dataloader:
            encoded_prompt = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            # This function applies the reverse watermark to the teacher model's logits
            reversed_watermarked_tokens = watermark.batch_generate_watermarked_tokens(encoded_prompt)
            decoded_texts = tokenizer.batch_decode(reversed_watermarked_tokens, skip_special_tokens=True)
            for idx in range(len(batch_prompts)):
                f.write(json.dumps({
                    "prompt": batch_prompts[idx],
                    "predict": decoded_texts[idx]
                }) + "\n")
                f.flush()
            pbar.update(1)
        
        pbar.close()
    
    return temp_output

def main(args):
    # Load prompt data (e.g., c4_truncate.json)
    with open(args.data_path, 'r') as f:
        # Assuming the file contains a list of JSON objects
        try:
            data = [json.loads(line) for line in f]
        except json.JSONDecodeError:
            # Fallback for a single JSON array
            f.seek(0)
            data = json.load(f)

    if args.shuffle:
        random.seed(42)
        random.shuffle(data)
    
    # Extract prompts based on data_mode
    if args.data_mode == 'c4':
        prompts = [d['prompt'] for d in data]
    elif args.data_mode == 'mmw':
        prompts = [d['system_message'] + "\n" + d['user_message'] + "\nAnswer:" for d in data]
    elif args.data_mode == 'dolly_cw':
        prompts = ["Question: " + d['prompt'] + "\n" + "\nAnswer:" for d in data]
    elif args.data_mode == 'writingprompts':
        prompts = ["Continue the following story creatively: " + d['prompt'] + '\n' + "Story: " for d in data]

    else:
        if args.template == 'default':
            prompts = ["Human: " + d.get('instruction', '') + "\n" + d.get('input', '') + "\nAssistant:" for d in data]

    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]

    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    num_processes = min(args.num_processes, torch.cuda.device_count())
    if num_processes == 0:
        print("No CUDA devices found. Running on a single CPU process.")
        num_processes = 1

    prompts_per_process = len(prompts) // num_processes
    prompt_splits = [
        prompts[i:i + prompts_per_process] 
        for i in range(0, len(prompts), prompts_per_process)
    ]

    time_stamp = int(time.time())

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            future = executor.submit(
                process_batch,
                i,
                time_stamp,
                prompt_splits[i],
                args
            )
            futures.append(future)

        temp_files = [future.result() for future in futures]

    with open(output_file, 'w') as outfile:
        for temp_file in temp_files:
            with open(temp_file, 'r') as infile:
                outfile.write(infile.read())
            os.remove(temp_file)

    print(f"\nGeneration completed. Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply a student-derived reverse watermark to a teacher model.')
    # --- MODIFIED: This is the path to the TEST/PROMPT data ---
    parser.add_argument('--data_path', type=str, 
                        default='prompts/c4_truncate.json',
                        help='Path to input prompt data for testing.')
    # --- MODIFIED: Argument name changed for clarity ---
    parser.add_argument('--teacher_model_path', type=str,
                        required=True,
                        help='Path to the TEACHER model.')
    parser.add_argument('--data_mode', type=str, default='unsup')
    parser.add_argument('--template', type=str, default='default')
    parser.add_argument('--reverse_watermark_config', type=str, required=True,
                        help='Path to the reverse watermark config file derived from the student model.')
    parser.add_argument('--max_samples', type=int, default=-1, help='Maximum number of samples to process. -1 for all.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for parallel processing.')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_file', type=str, default='spoofing_output/teacher_neutralized.jsonl', help='Path to save output files.')
    parser.add_argument('--shuffle', type=int, default=1, help='Shuffle input data.')
    args = parser.parse_args()
    main(args)
