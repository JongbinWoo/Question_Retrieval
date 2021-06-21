# Question Retrieval

## 개요
![pipeline](./pipeline.png)
- Document Collection
    - AI hub, KorQuAD 1.0에 있는 question-answer pair 40만건을 저장
    - 밑에 소개 될 Bi-Encoder 모델로 미리 계산된 question의 embedding을 저장해놓는다.
    - FAISS를 이용한 빠른 탐색
- Bi-Encoder
    - Sentence-BERT에 소개된 siamise network 구조
    - Pre-trained BERT를 STS(Semantic Textual Similarity) dataset으로 fine-tuning
        - 사용한 데이터 셋: KorSTS, KLUE-sts
    - query의 embedding을 계산하고 Document Collection에서 유사도가 큰 top-k개를 retrieval해온다.
- Cross-Encoder
    - 두 문장을 [SEP] 로 구분한뒤 BERT의 input으로 넣는다.
    - Pre-trained BERT를 STS(Semantic Textual Similarity) dataset으로 fine-tuning
        - 사용한 데이터 셋: KLUE-sts
    - 앞에서 retrieval해온 top-k개를 각각 query와 함께 Cross-Encoder의 input으로 넣어주면서 두 문장의 유사도를 측정한다. 일종의 parapharse검사.

##  학습
- Bi-Encoder
```
python train_bi_encoder.py
```
- Cross-Encoder

```
python train_cross_encoder.py
```

## 사용
question_retrieval.ipynb
