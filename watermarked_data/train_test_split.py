import json
import random

def split_jsonl_samples(source_path, sample_path, remaining_path, num_samples):
    print("-> Process in JSONL format.")
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"ERROR: Cannot find '{source_path}'.")
        return

    if total_lines < num_samples:
        print(f"ERROR: The number of data points ({total_lines}) is less than the number of samples ({num_samples}).")
        return

    sample_indices = set(random.sample(range(total_lines), num_samples))
    
    samples_written = 0
    remaining_written = 0
    with open(source_path, 'r', encoding='utf-8') as f_source, \
         open(sample_path, 'w', encoding='utf-8') as f_sample, \
         open(remaining_path, 'w', encoding='utf-8') as f_remaining:

        for i, line in enumerate(f_source):
            if i in sample_indices:
                f_sample.write(line)
                samples_written += 1
            else:
                f_remaining.write(line)
                remaining_written += 1

    print("\n--- Process done! ---")
    print(f"Sampled file: '{sample_path}' ({samples_written})")
    print(f"The rest: '{remaining_path}' ({remaining_written})")

def split_json_array_samples(source_path, sample_path, remaining_path, num_samples):
    print("-> Process in standard JSON format.")
    
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not isinstance(data, list):
        print("ERROR: The top-level element of the JSON file is not an array (list).")
        return

    total_items = len(data)
    if total_items < num_samples:
        print(f"ERROR: The number of data points ({total_items}) is less than the number of samples ({num_samples}).")
        return

    random.shuffle(data)
    samples = data[:num_samples]
    remaining = data[num_samples:]

    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    with open(remaining_path, 'w', encoding='utf-8') as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    print("\n--- Process done! ---")
    print(f"Sampled file: '{sample_path}' ({len(samples)})")
    print(f"The rest: '{remaining_path}' ({len(remaining)})")

def auto_detect_and_split(
    source_path: str,
    sample_path: str,
    remaining_path: str,
    num_samples: int = 100
):
    print(f"Start analyzing '{source_path}'...")
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            first_char = ""
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    first_char = stripped_line[0]
                    break
    except FileNotFoundError:
        print(f"ERROR: Cannot find '{source_path}'.")
        return

    if first_char == '[':
        # json
        split_json_array_samples(source_path, sample_path, remaining_path, num_samples)
    else:
        # jsonl
        split_jsonl_samples(source_path, sample_path, remaining_path, num_samples)


# Input file path
SOURCE_FILE = 'synthid/llama3.1-8b/watermarked_dolly_cw.json' 

# Path to save
SAMPLE_FILE = './test_output/synthid_from_teacher_testsets/from_llama3.1-8b_dolly_cw_test.jsonl'
REMAINING_FILE = './training_data/synthid_from_teacher_trainsets/from_llama3.1-8b_dolly_cw_train.jsonl'

# Decide #samples to test
SAMPLE_COUNT = 100


if __name__ == "__main__":
    auto_detect_and_split(SOURCE_FILE, SAMPLE_FILE, REMAINING_FILE, SAMPLE_COUNT)