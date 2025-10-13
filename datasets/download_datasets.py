import json
from datasets import load_dataset
from tqdm.auto import tqdm # 진행률 표시를 위한 라이브러리

# 1. 데이터셋 로드
# 스트리밍 모드를 사용하면 전체 데이터를 다운로드하지 않고도 일부를 처리할 수 있어 효율적입니다.
print("데이터셋 스트리밍 시작...")
ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)

# 2. 저장할 샘플 수와 파일명 지정
num_samples_to_save = 20000
output_filename = "writingprompts.jsonl"

# 3. 데이터셋에서 2만 개를 파일로 저장
print(f"{output_filename} 파일로 저장을 시작합니다...")

# 데이터셋에서 원하는 개수만큼만 가져옵니다.
subset_ds = ds.take(num_samples_to_save)

with open(output_filename, "w", encoding="utf-8") as f:
    for example in tqdm(subset_ds, total=num_samples_to_save):
        # 각 데이터(dictionary)를 JSON 문자열로 변환
        json_string = json.dumps(example, ensure_ascii=False)
        # 파일에 한 줄씩 쓰기
        f.write(json_string + "\n")

print(f"\n{num_samples_to_save}개의 데이터가 '{output_filename}' 파일에 저장되었습니다.")