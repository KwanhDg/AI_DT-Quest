import numpy as np
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeRegressor

# Tạo dữ liệu giả lập cho 100 sinh viên
np.random.seed(42)  # Đặt seed để kết quả nhất quán
num_students = 100

data = {
    'Gender': np.random.choice([0, 1], num_students),  # 0: Nam, 1: Nữ
    'Parent_education': np.random.choice([0, 1], num_students),  # 0: Thấp, 1: Cao
    'Parent_income': np.random.choice([0, 1, 2], num_students),  # 0: Thấp, 1: Trung bình, 2: Cao
    'First_child': np.random.choice([0, 1], num_students),  # 0: Không, 1: Có
    'Working': np.random.choice([0, 1], num_students)  # 0: Không, 1: Có
}

df = pd.DataFrame(data)

# Giả lập actual_grade dựa trên các biến số
noise = np.random.normal(0, 5, num_students)  # Thêm nhiễu ngẫu nhiên
df['actual_grade'] = 50 + 10*df['Parent_education'] + 10*df['Parent_income'] - 5*df['Working'] + noise
df['actual_grade'] = df['actual_grade'].clip(0, 100)  # Giới hạn trong khoảng 0-100

# Xây dựng mô hình cây quyết định để dự đoán actual_grade
X = df[['Gender', 'Parent_education', 'Parent_income', 'First_child', 'Working']]
y = df['actual_grade']
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Dự đoán actual_grade cho 3 trường hợp đại diện
case_excellent = np.array([[1, 1, 2, 1, 0]])  # Nữ, Phụ huynh trình độ cao, Thu nhập cao, Con đầu lòng, Không làm việc
case_average = np.array([[0, 0, 1, 0, 1]])    # Nam, Phụ huynh trình độ thấp, Thu nhập trung bình, Không phải con đầu lòng, Có làm việc
case_poor = np.array([[0, 0, 0, 0, 1]])       # Nam, Phụ huynh trình độ thấp, Thu nhập thấp, Không phải con đầu lòng, Có làm việc

excellent_grade = model.predict(case_excellent)[0]
average_grade = model.predict(case_average)[0]
poor_grade = model.predict(case_poor)[0]

print(f"Điểm dự đoán - Xuất sắc: {excellent_grade:.2f}")
print(f"Điểm dự đoán - Trung bình: {average_grade:.2f}")
print(f"Điểm dự đoán - Yếu: {poor_grade:.2f}")

# Định nghĩa các mức độ khó và điểm số tương ứng
levels = {1: 5, 2: 10, 3: 15, 4: 20}

def adaptive_test(actual_grade):
    """
    Hàm thực hiện kiểm tra thích nghi.
    Input: actual_grade (điểm thực tế của học sinh, từ 0-100)
    Output: điểm cuối cùng, số câu hỏi đã hỏi, và quá trình kiểm tra
    """
    grade = 0  # Điểm hiện tại
    questions_asked = 0  # Số câu hỏi đã hỏi
    path = []  # Lưu lại quá trình kiểm tra

    # Vòng lặp 1: Hỏi 4 câu hỏi từ Level 1 đến Level 4
    for level in range(1, 5):
        questions_asked += 1
        # Mô phỏng câu trả lời: đúng nếu random < actual_grade/100 và actual_grade >= level * 20
        is_correct = np.random.random() < (actual_grade / 100) and actual_grade >= (level * 20)
        score = levels[level] if is_correct else 0  # Ghi điểm nếu trả lời đúng
        grade += score
        path.append({
            'question': questions_asked,
            'level': level,
            'score': score,
            'total_score': grade,
            'correct': is_correct
        })

    # Vòng lặp 2: Tiếp tục hỏi đến khi đạt 100 điểm hoặc tối đa 10 câu
    while grade < 100 and questions_asked < 10:
        # Chọn mức độ khó tiếp theo dựa trên điểm hiện tại
        if grade >= 40:
            next_level = 4 if np.random.random() < 0.7 else 3  # Ưu tiên Level 4 hoặc 3
        elif grade >= 20:
            next_level = 3 if np.random.random() < 0.7 else 2  # Ưu tiên Level 3 hoặc 2
        else:
            next_level = 1 if np.random.random() < 0.7 else 2  # Ưu tiên Level 1 hoặc 2
        
        questions_asked += 1
        # Mô phỏng câu trả lời
        is_correct = np.random.random() < (actual_grade / 100) and actual_grade >= (next_level * 20)
        score = levels[next_level] if is_correct else 0
        grade += score
        grade = min(grade, 100)  # Giới hạn điểm tối đa là 100
        path.append({
            'question': questions_asked,
            'level': next_level,
            'score': score,
            'total_score': grade,
            'correct': is_correct
        })

    return grade, questions_asked, path

# Kiểm tra với các trường hợp dự đoán
cases = [
    {'name': 'Xuất sắc', 'actual_grade': excellent_grade},
    {'name': 'Trung bình', 'actual_grade': average_grade},
    {'name': 'Yếu', 'actual_grade': poor_grade}
]

results = []
for case in cases:
    np.random.seed(42)  # Đặt seed để kết quả nhất quán
    final_grade, num_questions, path = adaptive_test(case['actual_grade'])
    results.append({
        'name': case['name'],
        'actual_grade': case['actual_grade'],
        'final_grade': final_grade,
        'questions': num_questions,
        'path': path
    })

# In kết quả kiểm tra
print("**Kết quả kiểm tra động:**")
for result in results:
    print(f"\nTrường hợp: {result['name']}")
    print(f"Điểm thực tế (dự đoán): {result['actual_grade']:.2f}")
    print(f"Điểm cuối cùng: {result['final_grade']}")
    print(f"Số câu hỏi: {result['questions']}")
    print("Quá trình kiểm tra:")
    for step in result['path']:
        print(f"Câu {step['question']} (Level {step['level']}): {'Đúng' if step['correct'] else 'Sai'} - Tổng điểm: {step['total_score']}")

# Vẽ cây tổng quát với tất cả các trường hợp có thể xảy ra trong vòng lặp 1
dot_general = graphviz.Digraph(comment='Cây Tổng Quát')

# Câu hỏi 1 (Level 1)
dot_general.node('Q1', 'Q1 (Level 1)')

# Câu hỏi 2 (Level 2)
dot_general.node('Q2_5', 'Q2 (Level 2)\n5/5', style='filled', fillcolor='lightblue')
dot_general.node('Q2_0', 'Q2 (Level 2)\n0/5', style='filled', fillcolor='lightblue')

dot_general.edge('Q1', 'Q2_5', label='Đúng (5)')
dot_general.edge('Q1', 'Q2_0', label='Sai (0)')

# Câu hỏi 3 (Level 3)
dot_general.node('Q3_15', 'Q3 (Level 3)\n15/15', style='filled', fillcolor='lightgreen')
dot_general.node('Q3_5', 'Q3 (Level 3)\n5/15', style='filled', fillcolor='lightgreen')
dot_general.node('Q3_10', 'Q3 (Level 3)\n10/10', style='filled', fillcolor='lightgreen')
dot_general.node('Q3_0', 'Q3 (Level 3)\n0/10', style='filled', fillcolor='lightgreen')

dot_general.edge('Q2_5', 'Q3_15', label='Đúng (10)')
dot_general.edge('Q2_5', 'Q3_5', label='Sai (0)')
dot_general.edge('Q2_0', 'Q3_10', label='Đúng (10)')
dot_general.edge('Q2_0', 'Q3_0', label='Sai (0)')

# Câu hỏi 4 (Level 4)
dot_general.node('Q4_35', 'Q4 (Level 4)\n35/35', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_15', 'Q4 (Level 4)\n15/35', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_25', 'Q4 (Level 4)\n25/25', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_5', 'Q4 (Level 4)\n5/25', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_30', 'Q4 (Level 4)\n30/30', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_10', 'Q4 (Level 4)\n10/30', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_20', 'Q4 (Level 4)\n20/20', style='filled', fillcolor='lightcoral')
dot_general.node('Q4_0', 'Q4 (Level 4)\n0/20', style='filled', fillcolor='lightcoral')

dot_general.edge('Q3_15', 'Q4_35', label='Đúng (20)')
dot_general.edge('Q3_15', 'Q4_15', label='Sai (0)')
dot_general.edge('Q3_5', 'Q4_25', label='Đúng (20)')
dot_general.edge('Q3_5', 'Q4_5', label='Sai (0)')
dot_general.edge('Q3_10', 'Q4_30', label='Đúng (20)')
dot_general.edge('Q3_10', 'Q4_10', label='Sai (0)')
dot_general.edge('Q3_0', 'Q4_20', label='Đúng (20)')
dot_general.edge('Q3_0', 'Q4_0', label='Sai (0)')

dot_general.render('adaptive_testing_general', format='png', cleanup=True)
print("\nCây tổng quát đã được lưu thành 'adaptive_testing_general.png'")

# Vẽ cây cho từng trường hợp với các nhánh khác làm mờ
for i, result in enumerate(results, 1):
    dot = graphviz.Digraph(comment=f'Cây Trường Hợp {result["name"]}')
    
    # Thêm các nhánh của cây tổng quát làm mờ
    dot.node('Q1', 'Q1 (Level 1)')
    dot.node('Q2_5', 'Q2 (Level 2)\n5/5', style='filled', fillcolor='lightblue')
    dot.node('Q2_0', 'Q2 (Level 1)\n0/5', style='filled', fillcolor='lightblue')

    dot.edge('Q1', 'Q2_5', label='Đúng (5)', color='gray', style='dashed')
    dot.edge('Q1', 'Q2_0', label='Sai (0)', color='gray', style='dashed')

    dot.node('Q3_15', 'Q3 (Level 3)\n15/15', style='filled', fillcolor='lightgreen')
    dot.node('Q3_5', 'Q3 (Level 3)\n5/15', style='filled', fillcolor='lightgreen')
    dot.node('Q3_10', 'Q3 (Level 3)\n10/10', style='filled', fillcolor='lightgreen')
    dot.node('Q3_0', 'Q3 (Level 3)\n0/10', style='filled', fillcolor='lightgreen')

    dot.edge('Q2_5', 'Q3_15', label='Đúng (10)', color='gray', style='dashed')
    dot.edge('Q2_5', 'Q3_5', label='Sai (0)', color='gray', style='dashed')
    dot.edge('Q2_0', 'Q3_10', label='Đúng (10)', color='gray', style='dashed')
    dot.edge('Q2_0', 'Q3_0', label='Sai (0)', color='gray', style='dashed')

    dot.node('Q4_35', 'Q4 (Level 4)\n35/35', style='filled', fillcolor='lightcoral')
    dot.node('Q4_15', 'Q4 (Level 4)\n15/35', style='filled', fillcolor='lightcoral')
    dot.node('Q4_25', 'Q4 (Level 4)\n25/25', style='filled', fillcolor='lightcoral')
    dot.node('Q4_5', 'Q4 (Level 4)\n5/25', style='filled', fillcolor='lightcoral')
    dot.node('Q4_30', 'Q4 (Level 4)\n30/30', style='filled', fillcolor='lightcoral')
    dot.node('Q4_10', 'Q4 (Level 4)\n10/30', style='filled', fillcolor='lightcoral')
    dot.node('Q4_20', 'Q4 (Level 4)\n20/20', style='filled', fillcolor='lightcoral')
    dot.node('Q4_0', 'Q4 (Level 4)\n0/20', style='filled', fillcolor='lightcoral')

    dot.edge('Q3_15', 'Q4_35', label='Đúng (20)', color='gray', style='dashed')
    dot.edge('Q3_15', 'Q4_15', label='Sai (0)', color='gray', style='dashed')
    dot.edge('Q3_5', 'Q4_25', label='Đúng (20)', color='gray', style='dashed')
    dot.edge('Q3_5', 'Q4_5', label='Sai (0)', color='gray', style='dashed')
    dot.edge('Q3_10', 'Q4_30', label='Đúng (20)', color='gray', style='dashed')
    dot.edge('Q3_10', 'Q4_10', label='Sai (0)', color='gray', style='dashed')
    dot.edge('Q3_0', 'Q4_20', label='Đúng (20)', color='gray', style='dashed')
    dot.edge('Q3_0', 'Q4_0', label='Sai (0)', color='gray', style='dashed')

    # Thêm lộ trình thực tế nổi bật
    current_score = 0
    for step in result['path']:
        q_num = step['question']
        level = step['level']
        score = step['score']
        total_score = step['total_score']
        correct = step['correct']
        
        node_id = f"Q{q_num}_{total_score}"
        parent_id = 'Q1' if q_num == 1 else f"Q{q_num-1}_{current_score}"
        
        dot.node(node_id, f"Q{q_num} (Level {level})\n{total_score}/{min(total_score, 100)}", 
                 style='filled', fillcolor='lightcoral' if q_num > 4 else 'lightblue' if q_num == 2 else 'lightgreen' if q_num == 3 else 'white')
        dot.edge(parent_id, node_id, label=f"{'Đúng' if correct else 'Sai'} ({score})", 
                 color='blue' if result['name'] == 'Xuất sắc' else 'green' if result['name'] == 'Trung bình' else 'orange', 
                 style='bold')
        
        current_score = total_score
    
    dot.render(f'adaptive_testing_case_{i}', format='png', cleanup=True)
    print(f"Cây trường hợp {result['name']} đã được lưu thành 'adaptive_testing_case_{i}.png'")