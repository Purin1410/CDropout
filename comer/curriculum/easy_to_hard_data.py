import re
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt
# Tải từ điển
def load_dictionary(dictionary_file):
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        valid_tokens = set(line.strip() for line in f)
    return valid_tokens

# Tải chú thích của ảnh
def load_captions(caption_file):
    captions = []
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            image_name, label = line.strip().split('\t', 1)
            captions.append((image_name, label))
    return captions

# Tính Complexity Score
def calculate_complexity(label, valid_tokens):
    # Tiêu chí 1: Số lượng ký tự
    length = len(label.split())

    # Tiêu chí 2: Đếm ký tự đặc biệt
    operators = len(re.findall(r'[+\-*/^]', label))  # Các phép toán cơ bản
    special_symbols = len(re.findall(r'\\[a-zA-Z]+', label))  # Các ký hiệu LaTeX

    # Tiêu chí 3: Mức độ lồng nhau (ngoặc hoặc chỉ số con)
    nested_levels = label.count('{') + label.count('[') + label.count('(') - \
                    (label.count('}') + label.count(']') + label.count(')'))

    # Tính điểm phức tạp
    complexity_score = length + (operators * 2) + (special_symbols * 2) + (nested_levels * 3)
    return complexity_score

# Tính Percentile Thresholds
def get_percentile_thresholds(scores):
    easy_threshold = np.percentile(scores, 50)
    medium_threshold = np.percentile(scores, 85)
    return easy_threshold, medium_threshold

# Phân loại biểu thức thành simple, medium, complex
def classify_expression(complexity_score, easy_threshold, medium_threshold):
    if complexity_score < easy_threshold:
        return "simple"
    elif easy_threshold <= complexity_score < medium_threshold:
        return "medium"
    else:
        return "complex"

# Phân loại và lưu vào tệp
def classify_and_save(caption_file, dictionary_file, output_dir):
    valid_tokens = load_dictionary(dictionary_file)
    captions = load_captions(caption_file)

    # **BƯỚC 1:** Tính Complexity Score cho toàn bộ dữ liệu
    complexity_scores = []
    for _, label in captions:
        score = calculate_complexity(label, valid_tokens)
        complexity_scores.append(score)

    # **BƯỚC 2:** Tính Percentile Thresholds
    easy_threshold, medium_threshold = get_percentile_thresholds(complexity_scores)
    print(f"Easy Threshold: {easy_threshold}")
    print(f"Medium Threshold: {medium_threshold}")


    plt.hist(complexity_scores, bins=30)
    plt.axvline(x=easy_threshold, color='g', linestyle='--', label='Easy Threshold')
    plt.axvline(x=medium_threshold, color='r', linestyle='--', label='Medium Threshold')
    plt.xlabel('Complexity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # **BƯỚC 3:** Phân loại dựa trên Percentile Thresholds
    classified_data = defaultdict(list)
    for image_name, label in captions:
        score = calculate_complexity(label, valid_tokens)
        category = classify_expression(score, easy_threshold, medium_threshold)
        classified_data[category].append((image_name, label))

    # **BƯỚC 4:** Lưu kết quả vào các file phân loại
    for category, data in classified_data.items():
        output_file = f"{output_dir}/{category}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_name, label in data:
                f.write(f"{image_name}\t{label}\n")
        print(f"Saved {len(data)} items to {output_file}")

# Đường dẫn file
caption_file = "/content/caption.txt"
dictionary_file = "/content/dictionary.txt"
output_dir = "classified_data"  # Thư mục lưu kết quả

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Thực hiện phân loại
classify_and_save(caption_file, dictionary_file, output_dir)
