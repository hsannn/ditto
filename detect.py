import os
import json
import numpy as np
import torch
import argparse
import concurrent.futures
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
import math


def calculate_perplexity(text, model, tokenizer, device="cuda"):
    
    if not text:
        return None
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    try:
        max_length = model.config.max_position_embeddings
    except AttributeError:
        max_length = 1024 
        print(f"Warning: model.config.max_position_embeddings not found. Defaulting to {max_length}.")

    stride = 512
    nlls = []

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            
            trg_len = end_loc - begin_loc
            if trg_len == 0:
                continue

            current_input_ids = input_ids[:, begin_loc:end_loc]
            target_ids = current_input_ids.clone()
            
            if begin_loc > 0:
                target_ids[:, : -stride] = -100
                
            outputs = model(current_input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)

    if not nlls:
        return None
        
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def process_gpu_batch(rank, args, lines_batch):
    device = f"cuda:{rank}"
    
    wm_tokenizer = AutoTokenizer.from_pretrained(args.wm_model_path, trust_remote_code=True)
    wm_model = AutoModel.from_pretrained(args.wm_model_path, trust_remote_code=True, 
                                      torch_dtype=torch.bfloat16, device_map=device)

    transformers_config = TransformersConfig(model=wm_model,
                                             tokenizer=wm_tokenizer,
                                             vocab_size=len(wm_tokenizer),
                                             device=device)
    
    if args.watermark == 'kgw':
        myWatermark = AutoWatermark.load('KGW', algorithm_config=args.config_file, transformers_config=transformers_config)
    elif args.watermark == 'unigram':
        transformers_config.vocab_size = 151552
        myWatermark = AutoWatermark.load('Unigram', algorithm_config=args.config_file, transformers_config=transformers_config)
    elif args.watermark == 'synthid':
        myWatermark = AutoWatermark.load('SynthID', algorithm_config=args.config_file, transformers_config=transformers_config)
    else:
        raise ValueError(f"Unknown watermark type: {args.watermark}")
    
    # Load PPL model
    ppl_model = None
    if args.ppl_model_path:
        ppl_model = AutoModelForCausalLM.from_pretrained(
            args.ppl_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model_path)
        ppl_model.eval()

    results = []
    pbar = tqdm(total=len(lines_batch), desc=f'GPU {rank}', position=rank)
    
    for line_idx, line in enumerate(lines_batch):
        data = json.loads(line)
        prompt_text = data.get('prompt', '')
        predict_text = data['predict']
        
        # tokenize 'predict' and detect watermark
        tokens = wm_tokenizer.encode(predict_text, add_special_tokens=False)
        if len(tokens) == 0:
            continue

        token_tensor = torch.LongTensor(tokens).to(device)
        detection_result = myWatermark.detect_watermark_tokens(token_tensor)

        p_value = detection_result['p_value']
        if isinstance(p_value, torch.Tensor):
            p_value = p_value.item()

        z_score = detection_result.get('score', None)
        if isinstance(z_score, torch.Tensor):
            z_score = z_score.item()

        # Calculate perplexity
        perplexity = None
        if ppl_model:
            perplexity = calculate_perplexity(predict_text, ppl_model, ppl_tokenizer, device)
            if perplexity is not None and (math.isinf(perplexity) or math.isnan(perplexity)):
                perplexity = None
        
        # save results
        result_dict = {
            "prompt_idx": data.get('prompt_idx', line_idx),
            "raw_prompt": prompt_text,
            "generated_text": predict_text,
            "detection_result": {"p": p_value, "z_score": z_score},
            "perplexity": perplexity
        }
        results.append(result_dict)
        pbar.update(1)
    
    pbar.close()
    return results

def detect_and_save_results(args):
    print("Step 1: Detecting watermarks and generating results file...")
    
    with open(args.input) as f:
        lines = f.readlines()
    if args.max_samples > 0:
        lines = lines[:args.max_samples]

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise SystemError("No CUDA-enabled GPUs found.")
        
    lines_per_gpu = len(lines) // world_size
    
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
        futures = []
        for rank in range(world_size):
            start_idx = rank * lines_per_gpu
            end_idx = start_idx + lines_per_gpu if rank != world_size - 1 else len(lines)
            lines_batch = lines[start_idx:end_idx]
            
            if not lines_batch:
                continue
            
            future = executor.submit(process_gpu_batch, rank, args, lines_batch)
            futures.append(future)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Collecting results from GPUs"):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"An error occurred in a subprocess: {e}")
                raise

    all_results = sorted(all_results, key=lambda x: x["prompt_idx"])
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_total_results.jsonl")
    with open(output_path, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Detection finished. Results saved to {output_path}")
    return output_path


def get_p_list(results):
    return [res["detection_result"]["p"] for res in results if "detection_result" in res]

def get_ppl_list(results):
    ppl_scores = [res["perplexity"] for res in results if res.get("perplexity") is not None]
    return ppl_scores


def run_evaluation(total_results_path):
    print("Step 2.2: Calculating final evaluation metrics...")
    
    with open(total_results_path, 'r') as f:
        all_results = [json.loads(line) for line in f if line.strip()]

    p_list = np.array(get_p_list(all_results))
    ppl_list = get_ppl_list(all_results)
    avg_perplexity = np.mean(ppl_list) if ppl_list else float('nan')

    evaluation_dict = {
        "Avg_Perplexity": avg_perplexity,
        "median_p": np.median(p_list)
    }

    fpr_list = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
    for fpr in fpr_list:
        tpr = np.mean(p_list < fpr)
        evaluation_dict[f"TPR@FPR={fpr}"] = tpr
    
    return evaluation_dict


def main():
    parser = argparse.ArgumentParser(description="Watermark Detection and Evaluation Pipeline")
    
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file with prompts and predictions')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save all results')
    parser.add_argument('--wm_model_path', type=str, default="/workspace/intern_ckpt/panleyi/glm-4-9b-chat", help='Path to the tokenizer')
    parser.add_argument('--ppl_model_path', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Path to the model for perplexity calculation.')
    parser.add_argument('--watermark', type=str, default='kgw', choices=['kgw', 'unigram', 'synthid'], help='Watermark algorithm')
    parser.add_argument('--config_file', type=str, default='config/KGW.json', help='Path to the algorithm config file')
    parser.add_argument('--max_samples', type=int, default=-1, help='Max number of samples to process (-1 for all)')
    parser.add_argument('--dataset', type=str, help='Just for naming output files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    
    total_results_path = os.path.join(args.output_dir, f"{args.dataset}_total_results.jsonl")
    if not os.path.exists(total_results_path):
        total_results_path = detect_and_save_results(args)
    else:
        print(f"Found existing results at {total_results_path}. Skipping detection and PPL calculation.")

        
    final_evaluation = run_evaluation(total_results_path)
    
    evaluation_path = os.path.join(args.output_dir, f"{args.dataset}_evaluation_results.json")
    with open(evaluation_path, "w") as f:
        json.dump(final_evaluation, f, indent=4)
        
    print("\n" + "="*50)
    print("Final Evaluation Results:")
    print(json.dumps(final_evaluation, indent=4))
    print(f"Full evaluation report saved to {evaluation_path}")
    print("="*50)


if __name__ == '__main__':
    main()