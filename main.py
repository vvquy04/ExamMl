import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import string
import re




# Hàm tiền xử lý trước văn bản (loại bỏ dấu câu, chuyển đổi sang chữ thường)
def preprocess_text(text):
    text = text.lower()  # chuyển đổi sang chữ thường
    text = re.sub(f'[{string.punctuation}]', '', text)  # Loại bỏ dấu câu
    return text

#//1. Tải dữ liệu và chia tập huấn luyện và tập kiểm tra
#//1.1 Tải dữ liệu
# load dữ liệu
file_path = 'TestReviews.csv'
df = pd.read_csv(file_path)

# Hiển thị vài dòng đầu của dữ liệu
print(df.head())

#//2. Tiền xử lý văn bản
# Xử lý văn bản cho cột "review"
df['review'] = df['review'].apply(preprocess_text)

#// 1.2 Chia tập dữ liệu
# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra(20%)
X = df['review'] # Cột chứ văn bản đánh gía sản phẩm
y = df['class'] # Cột chứa nhãn phân loại (1: tích cực, 0: tiêu cực)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiển thị kích thước của tập huấn luyện và tập kiểm tra
print("Kích thước tập huấn luyện:", len(X_train))
print("Kích thước tập kiểm tra:", len(X_test))

#// 3. Chuyển đổi văn bản
# Chuyển đổi dữ liệu văn bản thành dạng số bằng phương pháp TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Giới hạn 5000 features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train) # Chuyển đổi tập huấn luyện
X_test_tfidf = tfidf_vectorizer.transform(X_test) # Chuyển đổi tập kiểm tra

# Hiển thị kích thước ma trận TF-IDF
print("Kích thước ma trận TF-IDF của tập huấn luyện:", X_train_tfidf.shape)
print("Kích thước ma trận TF-IDF của tập kiểm tra:", X_test_tfidf.shape)

# Hiển thị một số giá trị TF-IDF của một văn bản đầu tiên trong tập huấn luyện
print("Vector TF-IDF của văn bản đầu tiên trong tập huấn luyện:\n", X_train_tfidf[0].toarray())

# Hiển thị các từ khóa tương ứng với các cột trong ma trận
print("Các từ khóa tương ứng:", tfidf_vectorizer.get_feature_names_out()[:10])  # Hiển thị 10 từ khóa đầu tiên


#//4. Xây dựng mô hình
# Xây dựng mô hình Naive bayeso
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)


#//5. Đánh giá mô hình bằng accuracy và f1_score
# Đưa ra dự đoán trên tập kiểm tra
y_pred = naive_bayes_model.predict(X_test_tfidf)

# Đánh giá mô hình qua accuracy ( độ chính xác ) và f1
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Hiển thị kết quả đánh giá
print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")



