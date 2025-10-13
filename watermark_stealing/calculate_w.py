import argparse
import json
import math
import os

def calculate_w(args):
    with open(args.freq_file, 'r') as f:
        freq_dict = json.load(f)
    
    # frequency가 임계값보다 큰 prefix를 필터링
    filtered_dict = {
        k: v for k, v in freq_dict.items() 
        if v['frequency'] > args.freq_threshold
    }
    
    # 모든 prefix 중에서 가장 높은 frequency
    f_max = max(item['frequency'] for item in filtered_dict.values())
    
    # 각 prefix의 가중치 계산
    weights = {}
    alpha = 0.3
    
    for prefix_key, prefix_info in filtered_dict.items():
        try:
            freq = prefix_info['frequency']
            # 논문 수식 10: 1 / (log(f(prefix))/log(f_max))^α
            weight = 1 / pow(math.log(freq) / math.log(f_max), alpha)
            # prefix와 weight 정보 저장
            weights[prefix_key] = {
                'prefix': prefix_info['prefix'],
                'decoded': prefix_info['decoded'],
                'frequency': freq,
                'weight': weight
            }
        except (ValueError, ZeroDivisionError):
            continue
    
    # 결과저장
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(weights, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq_file", type=str, required=True, help="The frequency file.")
    parser.add_argument("--freq_threshold", type=float, required=True, help="The frequency threshold.")
    parser.add_argument("--output_file", type=str, required=True, help="The output file.")
    args = parser.parse_args()

    calculate_w(args)