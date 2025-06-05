from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
from underthesea import word_tokenize
import pickle
import json
import random
import os

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI()

# CORS cho frontend Blazor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:7067"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- ĐỌC DỮ LIỆU --------------------
try:
    with open('datasetv2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    raise RuntimeError(f"Lỗi khi đọc datasetv2.json: {e}")

# Cập nhật metadata
data['metadata']['last_updated'] = "2025-05-30T09:36:00+07:00"

try:
    with open('faqs.json', 'r', encoding='utf-8') as f:
        faqs_data = json.load(f)
        if not isinstance(faqs_data, list):
            faqs_data = faqs_data.get('faqs', [])
        if "faqs" not in data:
            data["faqs"] = []
        data["faqs"].extend(faqs_data)
except Exception as e:
    raise RuntimeError(f"Lỗi khi đọc faqs.json: {e}")

# -------------------- TIỀN XỬ LÝ --------------------
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Chatbot intents
if 'chatbot' in data and 'intents' in data['chatbot']:
    for intent in data['chatbot']['intents']:
        for pattern in intent.get('patterns', []):
            word_list = word_tokenize(pattern.lower())
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

# FAQs
for faq in data.get('faqs', []):
    for question in faq.get('questions', []):
        if isinstance(question, str):  # Kiểm tra question là chuỗi
            word_list = word_tokenize(question.lower())
            words.extend(word_list)
            documents.append((word_list, faq['tag']))
            if faq['tag'] not in classes:
                classes.append(faq['tag'])

# Kiểm tra từ vựng và classes
if not words or not classes:
    raise RuntimeError("Không có từ vựng hoặc classes. Kiểm tra datasetv2.json và faqs.json.")

# Lemmatize và loại bỏ từ không cần thiết
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Lưu lại
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# -------------------- TẠO DỮ LIỆU HUẤN LUYỆN --------------------
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

if not training:
    raise RuntimeError("Không có dữ liệu huấn luyện. Kiểm tra datasetv2.json và faqs.json.")

random.shuffle(training)
train_x = np.array([x[0] for x in training], dtype=np.float32)
train_y = np.array([x[1] for x in training], dtype=np.float32)

# -------------------- TẠO MÔ HÌNH --------------------
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Huấn luyện nếu chưa có mô hình
if os.path.exists('chatbot_model.h5'):
    model.load_weights('chatbot_model.h5')
    logger.info("Đã tải mô hình từ chatbot_model.h5")
else:
    logger.warning("Chưa có chatbot_model.h5, đang huấn luyện mô hình mới.")
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, callbacks=[early_stopping])
    model.save('chatbot_model.h5')

# -------------------- CHATBOT FUNCTIONS --------------------
conversation_history = []

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Chatbot server đang hoạt động."}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        ints = predict_class(request.message)
        response = get_contextual_response(request.message, ints, data)
        return {"response": response}
    except Exception as e:
        logger.error(f"Lỗi xử lý chat: {e}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý câu hỏi.")

def clean_up_sentence(sentence):
    try:
        sentence_words = word_tokenize(sentence.lower())
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words
    except Exception as e:
        logger.error(f"Lỗi tokenizing: {e}")
        return []

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    if not sentence_words:
        logger.warning("Không có từ nào sau khi tokenize.")
        return np.zeros(len(words), dtype=np.float32)
    return np.array([1 if lemmatizer.lemmatize(w) in sentence_words else 0 for w in words], dtype=np.float32)

def predict_class(sentence):
    if not words:
        logger.error("Không có từ vựng để dự đoán.")
        return []
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    threshold = 0.25
    results = [{"intent": classes[i], "probability": str(prob)} for i, prob in enumerate(res) if prob > threshold]
    results.sort(key=lambda x: float(x["probability"]), reverse=True)
    return results

def get_contextual_response(user_input, intents_list, data):
    global conversation_history
    conversation_history.append({'user': user_input, 'bot': None})

    if intents_list:
        tag = intents_list[0]['intent']

        # Kiểm tra intents
        for intent in data.get('chatbot', {}).get('intents', []):
            if intent['tag'] == tag:
                response = random.choice(intent.get('responses', ["Xin lỗi, tôi chưa hiểu câu hỏi."]))
                conversation_history[-1]['bot'] = response
                return response

        # Kiểm tra FAQs
        for faq in data.get('faqs', []):
            if faq.get('tag') == tag:
                response = faq.get('answer', "Thông tin đang được cập nhật.")
                conversation_history[-1]['bot'] = response
                return response

        # Kiểm tra majors (nếu có)
        if 'admission' in data and 'majors' in data['admission']:
            for major in data['admission']['majors']:
                if major.get('chatbot_tag') == tag:
                    response = f"Thông tin ngành {major['name']} (mã {major['code']}):\n"
                    response += f"- Chỉ tiêu: {major['quota']}\n"
                    response += f"- Phương thức xét tuyển: "
                    methods = [m['method_id'] for m in major.get('admission_methods', [])]
                    response += ", ".join([data['admission']['methods'][m-1]['name'] for m in methods]) + "\n"
                    response += "- Tổ hợp xét tuyển:\n"
                    for combo in major.get('subject_combinations', []):
                        response += f"  + {' + '.join(combo['subjects'])}\n"
                    conversation_history[-1]['bot'] = response
                    return response

    # Kiểm tra ngữ cảnh từ lịch sử
    context_keywords = ['tổ hợp', 'chỉ tiêu', 'xét tuyển']
    if any(keyword in user_input.lower() for keyword in context_keywords) and 'admission' in data and 'majors' in data['admission']:
        for history in reversed(conversation_history[:-1]):
            for major in data['admission']['majors']:
                if major['name'].lower() in history['user'].lower() or major['code'].lower() in history['user'].lower():
                    if 'tổ hợp' in user_input.lower():
                        response = f"Tổ hợp xét tuyển cho ngành {major['name']}:\n"
                        for combo in major.get('subject_combinations', []):
                            response += f"- {' + '.join(combo['subjects'])}\n"
                        conversation_history[-1]['bot'] = response
                        return response
                    if 'chỉ tiêu' in user_input.lower():
                        response = f"Chỉ tiêu ngành {major['name']}: {major['quota']}"
                        conversation_history[-1]['bot'] = response
                        return response

    # Phản hồi mặc định
    fallback = random.choice(data.get('chatbot', {}).get('fallback_responses', ["Xin lỗi, tôi chưa hiểu câu hỏi."]))
    conversation_history[-1]['bot'] = fallback
    return fallback

# -------------------- CHẠY SERVER --------------------
if __name__ == "__main__":
    # Kiểm tra file SSL
    ssl_keyfile = "server.key" if os.path.exists("server.key") else None
    ssl_certfile = "server.crt" if os.path.exists("server.crt") else None

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile
    )
