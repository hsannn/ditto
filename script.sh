# model config
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-2-7b-hf
# meta-llama/Llama-3.1-8B-Instruct

# generate training data (this is for MMW Bookreport experiment)
python generate_training_data.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --total_samples 10000 \
    --num_workers 4 \
    --output_dir watermarked_data/kgw/llama3.1-8b \
    --batch_size 2 \
    --final_output syntheticQA_for_mmw.json \
    --watermark kgw \
    --config_file config/KGW.json

# teacher generates WATERMARKED data!
python generate_watermarked_data.py --data_path datasets/dolly_cw.jsonl \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --data_mode dolly_cw \
    --max_samples 10000 \
    --batch_size 2 \
    --num_processes 4 \
    --max_new_tokens 512 \
    --output_file watermarked_data/synthid/llama3.1-8b/watermarked_dolly_cw.json \
    --shuffle 1 \
    --watermark 1 \
    --watermark_type synthid \
    --config_file config/SynthID.json

# Split train/test sets
python watermarked_data/train_test_split.py

# filter out low-quality data & remove duplicates & convert format
python utils/filter.py \
    --input_file watermarked_data/kgw/llama3.1-8b/syntheticQA_for_mmw.json \
    --filtered_file watermarked_data/kgw/llama3.1-8b/syntheticQA_for_mmw_filtered.json \
    --output_file watermarked_data/kgw/llama3.1-8b/syntheticQA_for_mmw_final.json

# sft on training data
# move training data to LLaMA Factory
mkdir LLaMA-Factory/data/llama3.2-3b
cp training_data/from_llama3.2-3b_dolly_cw_train.jsonl LLaMA-Factory/data/from_llama3.2-3b_dolly_cw_train.jsonl
# add dataset info in LLaMA Factory & write yaml file in LLaMA-Factory/examples/train_full/
# train model
cd LLaMA-Factory
llamafactory-cli train examples/train_full/llama1_full_sft_1.yaml


# watermark stealing
# step 1. Generate prefix frequency statistics files for n=1 and n=2 (tokenizer <- student's)
echo "### Step 1: prefix analysis"
echo "start prerfix 1"
python utils/analyze_prefix_frequency.py \
    --tokenizer_path meta-llama/Llama-3.2-3B-Instruct \
    --input_file training_data/from_llama3.1-8b_dolly_cw_train.jsonl \
    --output_file data_analysis/llama3.1-8b/training_data_prefix_freq/dolly_cw/prefix_1.json \
    --prefix_length 1

echo "start prerfix 2"
python utils/analyze_prefix_frequency.py \
    --tokenizer_path meta-llama/Llama-3.2-3B-Instruct \
    --input_file training_data/from_llama3.2-3b_dolly_cw_train.jsonl \
    --output_file data_analysis/llama3.2-3b/training_data_prefix_freq/dolly_cw/prefix_2.json \
    --prefix_length 2

# step 2. Calculate dn for all prefixes with frequencies higher than 5e-5 for n=1 and n=2, inputting them into the model before and after training.
export PYTHONPATH="/your/root:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

echo "start prerfix 1"
python watermark_stealing/calculate_local.py \
    --freq_file data_analysis/llama3.2-3b/training_data_prefix_freq/dolly_cw/prefix_1.json \
    --model_before_training meta-llama/Llama-3.2-3B-Instruct \
    --model_after_training LLaMA-Factory/trained_students/llama3.2-1b/kgw_dolly_cw \
    --output_file data_analysis/llama3.2-3b/local/dolly_cw/n_1.json \
    --freq_threshold 5e-5 \
    --data_file training_data/from_llama3.2-3b_dolly_cw_train.jsonl \
    --num_workers 4 \
    --num_workers_logits 2

echo "start prefix 2"
python watermark_stealing/calculate_local.py \
    --freq_file data_analysis/llama3.2-3b/training_data_prefix_freq/dolly_cw/prefix_2.json \
    --model_before_training meta-llama/Llama-3.2-3B-Instruct \
    --model_after_training LLaMA-Factory/trained_students/llama3.2-1b/kgw_dolly_cw \
    --output_file data_analysis/llama3.2-3b/local/dolly_cw/n_2.json \
    --freq_threshold 5e-5 \
    --data_file training_data/from_llama3.2-3b_dolly_cw_train.jsonl \
    --num_workers 4 \
    --num_workers_logits 2

## step 3. Calculate d0
python watermark_stealing/calculate_global.py \
    --model_before_training meta-llama/Llama-3.2-3B-Instruct \
    --model_after_training LLaMA-Factory/trained_students/llama3.2-1b/kgw_dolly_cw \
    --output_file data_analysis/llama3.2-3b/local/dolly_cw/n_0.json \
    --data_file training_data/from_llama3.2-3b_dolly_cw_train.jsonl \
    --num_workers_logits 2

## step 4. Calculate weight for each prefix based on its frequency in training data
echo "Step 4: calculate weights"
### n = 1
echo "### start w_1"
python watermark_stealing/calculate_w.py \
    --freq_file data_analysis/llama3.2-3b/training_data_prefix_freq/dolly_cw/prefix_1.json \
    --freq_threshold 5e-5 \
    --output_file data_analysis/llama3.2-3b/prefix_weight/dolly_cw/w_1.json

### n = 2
echo "### start w_2"
python watermark_stealing/calculate_w.py \
    --freq_file data_analysis/llama3.2-3b/training_data_prefix_freq/dolly_cw/prefix_2.json \
    --freq_threshold 5e-5 \
    --output_file data_analysis/llama3.2-3b/prefix_weight/dolly_cw/w_2.json

# Imitate teacher model's watermark. Check "config/ReverseWatermark.json" first.
python do_spoofing.py \
    --data_path test_output/from_llama3.1-8b_dolly_cw_test.jsonl \
    --teacher_model_path meta-llama/Llama-3.2-3B-Instruct \
    --data_mode dolly_cw \
    --reverse_watermark_config config/ReverseWatermark.json \
    --output_file spoofing_output/llama3.1-8b/dolly_cw/delta5_spoofed_output.jsonl \
    --max_samples 100 \
    --batch_size 2 \
    --num_processes 4


python detect.py \
    --input spoofing_output/llama3.1-8b/dolly_cw/delta5_spoofed_output.jsonl \
    --output_dir ./exp_results/spoofing/dolly_cw/delta5_mimic_llama3.1-8b \
    --wm_model_path meta-llama/Llama-3.2-3B-Instruct \
    --ppl_model_path meta-llama/Llama-2-7b-chat-hf \
    --dataset dolly_cw \
    --watermark kgw \
    --config_file config/KGW.json \
