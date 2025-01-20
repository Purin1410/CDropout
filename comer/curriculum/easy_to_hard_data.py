import re
from collections import defaultdict

def load_dictionary(dictionary_file):
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        valid_tokens = set(line.strip() for line in f)
    return valid_tokens

def load_captions(caption_file):
    captions = []
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            image_name, label = line.strip().split('\t', 1)
            captions.append((image_name, label))
    return captions

def calculate_complexity(label, valid_tokens):
    # Tiêu chí 1: Số lượng ký tự
    length = len(label.split())

    # Tiêu chí 2: Đếm ký tự đặc biệt
    operators = len(re.findall(r'[+\-*/^]', label))  # Các phép toán cơ bản
    special_symbols = len(re.findall(r'\\[a-zA-Z]+', label))  # Các ký hiệu LaTeX

    # Tiêu chí 3: Mức độ lồng nhau (ngoặc hoặc chỉ số con)
    nested_levels = label.count('{') + label.count('[') + label.count('(') - \
                    (label.count('}') + label.count(']') + label.count(')'))

    # Kiểm tra số ký tự có trong từ điển
    tokens = label.split()
    valid_token_count = sum(1 for token in tokens if token in valid_tokens)

    # Tính điểm phức tạp
    complexity_score = length + (operators * 2) + (special_symbols * 3) + nested_levels - valid_token_count
    return complexity_score

# Phân loại biểu thức thành simple, medium, complex
def classify_expression(label, valid_tokens):
    complexity_score = calculate_complexity(label, valid_tokens)

    if complexity_score <= 5:
        return "simple"
    elif 5 < complexity_score <= 15:
        return "medium"
    else:
        return "complex"

# Phân loại và lưu vào tệp
def classify_and_save(caption_file, dictionary_file, output_dir):
    valid_tokens = load_dictionary(dictionary_file)
    captions = load_captions(caption_file)

    # Khởi tạo các danh sách cho các loại
    classified_data = defaultdict(list)

    # Phân loại từng nhãn
    for image_name, label in captions:
        category = classify_expression(label, valid_tokens)
        classified_data[category].append((image_name, label))

    # Lưu kết quả vào các file phân loại
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
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Thực hiện phân loại
classify_and_save(caption_file, dictionary_file, output_dir)
