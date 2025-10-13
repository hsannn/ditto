import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import argparse

# === 워터마크 import 복원 ===
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

# ==============================================================================
# === Mode 1: Self-Instruct (모델이 Q/A 모두 생성)를 위한 함수들 ===
# ==============================================================================

def generate_self_instruct_batch(model, tokenizer, instruction, batch_size, watermark=None):
    instructions = [instruction] * batch_size
    inputs = tokenizer(instructions, return_tensors="pt").to(model.device)
    
    # === 워터마크 로직 복원 ===
    if watermark:
        outputs = watermark.batch_generate_watermarked_tokens(inputs)
    else:
        outputs = model.generate(
            **inputs, max_new_tokens=2000, do_sample=True, temperature=0.7, top_p=0.9
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    filtered_responses = []
    # (파싱 로직은 이전과 동일)
    for response in responses:
        parts = response.split("[[")
        parts = [r.split("]]")[0] for r in parts]
        for part in parts:
            if "Instruction:" in part and "Input:" in part and "Answer:" in part:
                instruction = part.split("Instruction:")[1].split("Input:")[0].strip()
                input_text = part.split("Input:")[1].split("Answer:")[0].strip()
                answer = part.split("Answer:")[1].strip()
                filtered_responses.append({
                    "Instruction": instruction, "Input": input_text, "Answer": answer
                })
    return filtered_responses

def worker_self_instruct(rank, t, instruction, model_path, output_dir, samples_per_worker, batch_size, watermark_type, config_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    # === 워터마크 로직 복원 ===
    myWatermark = None
    if watermark_type:
        transformers_config = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer), device=model.device, max_new_tokens=2000, do_sample=True, temperature=0.7, top_p=0.9)
        if watermark_type == "kgw":
            myWatermark = AutoWatermark.load('KGW', algorithm_config=config_file, transformers_config=transformers_config)
        elif watermark_type == "synthid":
            myWatermark = AutoWatermark.load('SynthID', algorithm_config=config_file, transformers_config=transformers_config)
        elif watermark_type == "unigram":
            transformers_config.vocab_size = 151552  # 예시 vocab_size
            myWatermark = AutoWatermark.load('Unigram', algorithm_config=config_file, transformers_config=transformers_config)

    output_file = os.path.join(output_dir, f"worker_{rank}_{t}.json")
    pbar = tqdm(total=samples_per_worker, position=rank, desc=f"Worker {rank} [Self-Instruct]")
    
    generated_samples = 0
    with open(output_file, 'w', encoding='utf-8') as file:
        while generated_samples < samples_per_worker:
            batch = generate_self_instruct_batch(model, tokenizer, instruction, batch_size, watermark=myWatermark) # watermark 객체 전달
            for response in batch:
                if generated_samples < samples_per_worker:
                    file.write(json.dumps(response, ensure_ascii=False))
                    file.write('\n')
                    generated_samples += 1
                    pbar.update(1)
                else:
                    break
            file.flush()
    pbar.close()
    return output_file

# ==============================================================================
# === Mode 2: Answer Generation (주어진 Q에 A만 생성)을 위한 함수들 ===
# ==============================================================================

def generate_answers_batch(model, tokenizer, batch_prompts, watermark=None):
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
    
    # === 워터마크 로직 복원 ===
    if watermark:
        outputs = watermark.batch_generate_watermarked_tokens(inputs)
    else:
        outputs = model.generate(
            **inputs, max_new_tokens=2000, do_sample=True, temperature=0.7, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    completed_responses = []
    # (결과 저장 로직은 이전과 동일)
    for i, full_text in enumerate(decoded_outputs):
        prompt_len = len(batch_prompts[i])
        answer_text = full_text[prompt_len:].strip()
        completed_responses.append({
            "prompt": batch_prompts[i], "generated_answer": answer_text
        })
    return completed_responses

def worker_answer_gen(rank, t, worker_prompts, model_path, output_dir, batch_size, watermark_type, config_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    # === 워터마크 로직 복원 ===
    myWatermark = None
    if watermark_type:
        transformers_config = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=len(tokenizer), device=model.device, max_new_tokens=2000, do_sample=True, temperature=0.7, top_p=0.9)
        if watermark_type == "kgw":
            myWatermark = AutoWatermark.load('KGW', algorithm_config=config_file, transformers_config=transformers_config)
        elif watermark_type == "synthid":
            myWatermark = AutoWatermark.load('SynthID', algorithm_config=config_file, transformers_config=transformers_config)
        elif watermark_type == "unigram":
            transformers_config.vocab_size = 151552  # 예시 vocab_size
            myWatermark = AutoWatermark.load('Unigram', algorithm_config=config_file, transformers_config=transformers_config)

    output_file = os.path.join(output_dir, f"worker_{rank}_{t}.json")
    total_samples = len(worker_prompts)
    pbar = tqdm(total=total_samples, position=rank, desc=f"Worker {rank} [Answer-Gen]")
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for i in range(0, total_samples, batch_size):
            batch_chunk = worker_prompts[i:i + batch_size]
            if not batch_chunk: continue
            generated_batch = generate_answers_batch(model, tokenizer, batch_chunk, watermark=myWatermark) # watermark 객체 전달
            for response in generated_batch:
                file.write(json.dumps(response, ensure_ascii=False))
                file.write('\n')
            file.flush()
            pbar.update(len(batch_chunk))
    pbar.close()
    return output_file

# ==============================================================================
# === 공통 함수 및 메인 실행 로직 (이전과 동일) ===
# ==============================================================================
def merge_files(output_files, final_output):
    with open(final_output, 'w', encoding='utf-8') as outfile:
        for filename in output_files:
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # (argparse 설정은 이전과 동일)
    parser.add_argument("--model_path", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--dataset", type=str, default=None, choices=["c4", "mmw", "dolly_cw"], help="[Answer-Gen Mode] Name of the dataset to use.")
    parser.add_argument("--total_samples", type=int, default=100, help="[Self-Instruct Mode] Total number of samples to generate.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default='training_data')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--final_output", type=str, default='generated_data.jsonl')
    parser.add_argument("--watermark", type=str, default=None, choices=['kgw', 'synthid', 'unigram'], help="Watermark type.")
    parser.add_argument("--config_file", type=str, default='config/KGW.json', help="Path to watermark config file.")
    args = parser.parse_args()

    # (main 함수 로직은 이전과 동일)
    os.makedirs(args.output_dir, exist_ok=True)
    final_output = os.path.join(args.output_dir, args.final_output)
    mp.set_start_method('spawn', force=True)
    t = int(time.time())

    if args.dataset:
        # ... (Answer-Gen 모드 로직)
        print(f"Mode: Answer Generation from '{args.dataset}' dataset.")
        all_prompts = []
        if args.dataset == "c4":
            with open("datasets/c4_truncate.json", "r") as f: data = json.load(f)
            all_prompts = [d['text'] for d in data]
        elif args.dataset == "mmw":
            with open("datasets/mmw_bookreport.json", "r") as f: data = json.load(f)
            all_prompts = [f"{d.get('system_message', '')}\n{d.get('user_message', '')}\nAnswer:" for d in data]
        elif args.dataset == "dolly_cw":
            with open("datasets/dolly_cw.json", "r") as f: data = json.load(f)
            all_prompts = [f"Question: {d.get('instruction', '')}\nContext: {d.get('context', '')}\nAnswer:" for d in data]
        print(f"Total prompts to process: {len(all_prompts)}")
        chunks = [all_prompts[i::args.num_workers] for i in range(args.num_workers)]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(worker_answer_gen, i, t, chunks[i], args.model_path, args.output_dir, args.batch_size, args.watermark, args.config_file) for i in range(args.num_workers)]
            output_files = [future.result() for future in futures]
    else:
        # ... (Self-Instruct 모드 로직)
        print("Mode: Self-Instruction (LLM is making both Questions and Answers).")
        print(f"Total samples to generate: {args.total_samples}")
        with open("template/instruction.txt", "r", encoding="utf-8") as file:
            instruction = file.read().strip()
        samples_per_worker = args.total_samples // args.num_workers
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(worker_self_instruct, i, t, instruction, args.model_path, args.output_dir, samples_per_worker, args.batch_size, args.watermark, args.config_file) for i in range(args.num_workers)]
            output_files = [future.result() for future in futures]

    merge_files(output_files, final_output)
    for file in output_files:
        os.remove(file)
    print(f"Generation complete. Results are in {final_output}")