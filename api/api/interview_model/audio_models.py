import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import librosa

# 1. 필요한 모델 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# 2. 신경망 모델 정의
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# 3. AudioEmotionAnalysModel 클래스 정의
class AudioEmotionAnalysModel:
    def __init__(self):
        # 감정 범주 정의
        self.emotions = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        output_dim = len(self.emotions)

        # 신경망 모델 및 손실 함수, 최적화기 설정
        input_dim = 768 * 2  # Wav2Vec2와 BERT 특징 벡터 크기 결합
        hidden_dim = 256
        self.model = EmotionClassifier(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # 음성 특징 추출 함수
    def extract_audio_features(self, audio_path):
        audio_input, _ = librosa.load(audio_path, sr=16000)
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            audio_features = wav2vec2_model(inputs.input_values).last_hidden_state
        return audio_features.mean(dim=1)  # 평균 pooling

    # 텍스트 특징 추출 함수
    def extract_text_features(self, text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = bert_model(**inputs).last_hidden_state
        return text_features.mean(dim=1)  # 평균 pooling

    # 감정 분석 및 분류 함수
    def analyze_emotion(self, audio_path, text):
        audio_features = self.extract_audio_features(audio_path)
        text_features = self.extract_text_features(text)
        
        # 멀티모달 특징 결합
        combined_features = torch.cat((audio_features, text_features), dim=1)

        # 신경망 모델을 통해 감정 예측
        with torch.no_grad():
            outputs = self.model(combined_features)
            prediction = torch.argmax(outputs, dim=1).item()

        return self.emotions[prediction]
if __name__ == "__main__":
    # 사용 예제
    analys_model = AudioEmotionAnalysModel()
    audio_path = "ckmk_a_mm_m_n_140614.wav"
    text = "제가 다른 지원자들에 비해 그나마 조금 더 다른 차변성을 말씀드리자면 저는 경험이라고 생각을 합니다. 저는 동업정 동군의 일을 약 6개월 정도 인턴으로 근무해 본 적이 있습니다. 지금 이 회사는 아니지만 그 회사에서 약 6개월 동안 인턴으로 근무를 해 본 적이 있고 그 기간 동안 꽤 높은 좋은 평가를 받았던 것으로 기억을 합니다. 그 경험 말고도 저는 생산라인 공정에서 아르바이트를 해 본 경험이 꽤 많습니다. 저는 이력서를 보시면 아시겠지만 약 한 전전 같은 데서는 거의 한 10개월 정도 경험을 했었고 그때 아르바이트였지만 그래도 그 부분, 그 라인의 작업 반장까지 맡아서 할 정도로 저는 경험과 노하우를 가지고 있습니다. 제가 비록 내세울 것은 경험밖에 없지만 이 일을 함에 있어서 경험은 매우 중요하다고 저는 생각을 합니다. 이 라인에서 일을 공정을 할 때 이 기계가 어떻게 돌아가고 최소한 어떻게 작동하는지 정도는 알고 있어야 저는 안전하다고 생각하기 때문입니다."
    emotion = analys_model.analyze_emotion(audio_path, text)
    print(f"Predicted Sentiment Score: {emotion}")