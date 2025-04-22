import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import graphviz

# Tạo dữ liệu giả lập
def generate_student_data(num_students=100):
    np.random.seed(42)
    data = {
        'Gender': np.random.choice([0, 1], num_students),
        'Parent_education': np.random.choice([0, 1], num_students),
        'Parent_income': np.random.choice([0, 1, 2], num_students)
    }
    df = pd.DataFrame(data)
    df['actual_grade'] = 50 + 10*df['Parent_education'] + 10*df['Parent_income'] + np.random.normal(0, 5, num_students)
    return df

# Huấn luyện mô hình dự đoán
def train_predictive_model(df):
    X = df.drop('actual_grade', axis=1)
    y = df['actual_grade']
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model

# Vẽ cây quyết định biểu diễn lộ trình kiểm tra
def visualize_test_path(path, filename="test_path"):
    dot = graphviz.Digraph(comment="Test Path", format="png")
    dot.node("start", "Bắt đầu", shape="ellipse", style="filled", fillcolor="lightblue")

    for i, step in enumerate(path):
        question_id = f"Q{i+1}"
        question_label = f"Câu hỏi {step['question']}\nLevel {step['level']}\nĐiểm: {step['score']}"
        dot.node(question_id, question_label, shape="box", style="filled", fillcolor="lightgreen" if step['score'] > 0 else "lightcoral")
        
        # Liên kết các câu hỏi
        if i == 0:
            dot.edge("start", question_id)
        else:
            dot.edge(f"Q{i}", question_id)

        # Thêm thông tin điểm tổng
        if i == len(path) - 1:
            end_label = f"Kết thúc\nTổng điểm: {step['total']}"
            dot.node("end", end_label, shape="ellipse", style="filled", fillcolor="lightblue")
            dot.edge(question_id, "end")

    # Lưu và hiển thị cây
    dot.render(filename, view=True)
    return dot

# Kiểm tra thích nghi với tương tác và dự đoán ban đầu
def adaptive_test_hybrid(student_data, model, max_questions=10, target_score=100):
    # Dự đoán khả năng ban đầu
    actual_grade = model.predict(student_data)[0]
    print(f"Dự đoán khả năng ban đầu: {actual_grade:.2f}/100")

    levels = {1: 5, 2: 10, 3: 15, 4: 20}
    current_score = 0
    questions_asked = 0
    path = []

    # Vòng 1: 4 câu hỏi cố định
    for level in range(1, 5):
        questions_asked += 1
        print(f"Câu hỏi {questions_asked} (Level {level}, {levels[level]} điểm): Đây là câu hỏi giả lập. Nhập '1' nếu đúng, '0' nếu sai.")
        answer = input("Câu trả lời của bạn: ")
        is_correct = int(answer) == 1
        score = levels[level] if is_correct else 0
        current_score += score
        path.append({'question': questions_asked, 'level': level, 'score': score, 'total': current_score})
        if questions_asked >= max_questions or current_score >= target_score:
            break

    # Vòng 2: Điều chỉnh độ khó
    while current_score < target_score and questions_asked < max_questions:
        next_level = 4 if current_score >= 40 else 3 if current_score >= 20 else 1
        questions_asked += 1
        print(f"Câu hỏi {questions_asked} (Level {next_level}, {levels[next_level]} điểm): Đây là câu hỏi giả lập. Nhập '1' nếu đúng, '0' nếu sai.")
        answer = input("Câu trả lời của bạn: ")
        is_correct = int(answer) == 1
        score = levels[next_level] if is_correct else 0
        current_score += score
        path.append({'question': questions_asked, 'level': next_level, 'score': score, 'total': current_score})

    # Vẽ cây quyết định
    visualize_test_path(path, "test_path_visualization")

    return current_score, questions_asked, path

# Chạy thử nghiệm
if __name__ == "__main__":
    df = generate_student_data()
    model = train_predictive_model(df)
    example_student = np.array([[1, 1, 2]])  # Nữ, phụ huynh trình độ cao, thu nhập cao
    final_score, num_questions, path = adaptive_test_hybrid(example_student, model)
    print(f"\nKết quả cuối cùng:")
    print(f"Điểm cuối: {final_score}, Số câu hỏi: {num_questions}")
    print("Lộ trình kiểm tra:", path)