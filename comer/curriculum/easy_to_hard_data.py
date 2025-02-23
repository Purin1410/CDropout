import re
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt

# Tải chú thích của ảnh
def load_captions(caption_file):
    captions = []
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            image_name, label = line.strip().split('\t', 1)
            captions.append((image_name, label))
    return captions


# Định nghĩa các nhóm ký hiệu đặc biệt theo phân chia
easy_symbols = [
    r'\Delta', r'\Pi', r'\alpha', r'\beta', r'\gamma',
    r'\lambda', r'\mu', r'\phi', r'\pi', r'\sigma', r'\theta', r'\in'
]

medium_symbols = [
    r'\lim', r'\log', r'\cos', r'\sin', r'\tan',
    r'\cdot', r'\div', r'\cdots', r'\ldots', r'\times', r'\pm', r'\prime'
]

hard_symbols = [
    r'\frac', r'\sqrt', r'\sum', r'\int', r'\rightarrow',
    r'\forall', r'\exists', r'\neq', r'\geq', r'\leq', r'\infty', r'\limits'
]

def compute_special_symbol_scores(label):
    easy_count = 0
    medium_count = 0
    hard_count = 0

    for sym in easy_symbols:
        easy_count += label.count(sym)
    for sym in medium_symbols:
        medium_count += label.count(sym)
    for sym in hard_symbols:
        hard_count += label.count(sym)

    return easy_count, medium_count, hard_count

def max_nested_depth(label):
    """Tính độ sâu lồng ghép tối đa của các dấu ngoặc"""
    max_depth = 0
    current_depth = 0
    for char in label:
        if char in "({[":
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char in ")}]":
            if current_depth > 0:
                current_depth -= 1
    return max_depth

def extract_features(label):
    # 1. Độ dài: số từ (có thể sử dụng số ký tự nếu cần)
    length = len(label.split())

    # 2. Phép toán cơ bản: đếm các toán tử +, -, *, /
    operators = len(re.findall(r'[+\-*/]', label))

    # 4. Độ lồng ghép: tính độ sâu lồng nhau tối đa của các dấu ngoặc
    nested_depth = max_nested_depth(label)

    # 5. Chỉ số dưới/chỉ số trên: đếm "_" và "^"
    sub_sup_count = label.count('_') + label.count('^')

    # 7. Tính điểm cho các ký hiệu đặc biệt theo nhóm (easy, medium, hard)
    simple_symbol, medium_symbol, complex_symbol = compute_special_symbol_scores(label)

    return length, operators, simple_symbol, medium_symbol, complex_symbol, nested_depth, sub_sup_count

# Chuẩn hóa dữ liệu theo Min-Max Scaling
def min_max_scaling(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:  # Tránh chia cho 0 nếu mọi giá trị giống nhau
        return [0] * len(values)
    return [(x - min_val) / (max_val - min_val) for x in values]

# Tính Complexity Score với dữ liệu đã chuẩn hóa
def calculate_complexity(
    scaled_length, scaled_operators, scaled_simple_symbols, scaled_medium_symbols, scaled_complex_symbols,scaled_nested_levels, scaled_sub_sup_counts):
    return 0.05*scaled_length + 0.1*scaled_simple_symbols + 0.1*scaled_operators + 0.15*scaled_medium_symbols + 0.2*scaled_complex_symbols+ 0.25*scaled_nested_levels + 0.15*scaled_sub_sup_counts

# Tính Percentile Thresholds
def get_percentile_thresholds(scores):
    easy_threshold = np.percentile(scores, 40)
    medium_threshold = np.percentile(scores, 80)
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
def classify_and_save(caption_file, output_dir):
    captions = load_captions(caption_file)

    # **BƯỚC 1:** Trích xuất đặc trưng cho toàn bộ dữ liệu
    lengths, operators, simple_symbols, medium_symbols, complex_symbols, nested_depths, sub_sup_counts = [], [], [], [], [], [], []
    for _, label in captions:
        # length, operators, simple_symbol, medium_symbol, complex_symbol, nested_depth, sub_sup_count
        l, o, s_sym, m_sym, c_sym, n, sub = extract_features(label)
        lengths.append(l)
        operators.append(o)
        simple_symbols.append(s_sym)
        medium_symbols.append(m_sym)
        complex_symbols.append(c_sym)
        nested_depths.append(n)
        sub_sup_counts.append(sub)

    # **BƯỚC 2:** Chuẩn hóa dữ liệu bằng Min-Max Scaling
    scaled_lengths = min_max_scaling(lengths)
    scaled_operators = min_max_scaling(operators)
    scaled_simple_symbols = min_max_scaling(simple_symbols)
    scaled_medium_symbols = min_max_scaling(medium_symbols)
    scaled_complex_symbols = min_max_scaling(complex_symbols)
    scaled_nested_levels = min_max_scaling(nested_depths)
    scaled_sub_sup_counts = min_max_scaling(sub_sup_counts)

    # **BƯỚC 3:** Tính Complexity Score từ dữ liệu đã chuẩn hóa
    complexity_scores = []
    for i in range(len(captions)):
        score = calculate_complexity(
            scaled_lengths[i],
            scaled_operators[i],
            scaled_simple_symbols[i],
            scaled_medium_symbols[i],
            scaled_complex_symbols[i],
            scaled_nested_levels[i],
            scaled_sub_sup_counts[i],
        )
        complexity_scores.append(score)

    # features_dict = { 'Scaled Length': scaled_lengths, 'Scaled Operators': scaled_operators, 'Scaled Simple Symbols': scaled_simple_symbols, 'Scaled Medium Symbols': scaled_medium_symbols, 'Scaled Complex Symbols': scaled_complex_symbols, 'Scaled Nested Levels': scaled_nested_levels, 'Scaled Sub/Sup Counts': scaled_sub_sup_counts }

    # num_features = len(features_dict)
    # plt.figure(figsize=(18, 12))
    # for i, (feature_name, data) in enumerate(features_dict.items(), start=1):
    #   plt.subplot(3, 3, i)
    #   plt.hist(data, bins=30, alpha=0.7, color='blue')
    #   plt.title(feature_name)
    #   plt.xlabel('Value')
    #   plt.ylabel('Frequency')


    # **BƯỚC 4:** Xác định ngưỡng phân loại
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

    # **BƯỚC 5:** Phân loại dựa trên Percentile Thresholds
    classified_data = defaultdict(list)
    for i, (image_name, label) in enumerate(captions):
        category = classify_expression(complexity_scores[i], easy_threshold, medium_threshold)
        classified_data[category].append((image_name, label))

    # **BƯỚC 6:** Lưu kết quả vào các file phân loại
    for category, data in classified_data.items():
        output_file = f"{output_dir}/{category}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_name, label in data:
                f.write(f"{image_name}\t{label}\n")
        print(f"Saved {len(data)} items to {output_file}")

# Đường dẫn file
caption_file = "/content/caption.txt"
output_dir = "classified_data"  # Thư mục lưu kết quả

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Thực hiện phân loại
classify_and_save(caption_file, output_dir)
