import json
from sentence_transformers.readers import InputExample

def load_sts_dataset():
    klue_train_path = '/content/klue-sts-v1/klue-sts-v1_train.json'
    klue_dev_path = '/content/klue-sts-v1/klue-sts-v1_dev.json'
    korsts_train_path = '/content/drive/MyDrive/sentence-bert/KorSTS/sts-train.tsv'
    korsts_dev_path = '/content/drive/MyDrive/sentence-bert/KorSTS/sts-dev.tsv'
    korsts_test_path = '/content/drive/MyDrive/sentence-bert/KorSTS/sts-test.tsv'

    train_samples = []
    dev_samples = []
    test_samples = []

    with open(klue_train_path, 'rt', encoding='utf-8') as fIn:
        klue_train = json.load(fIn)

    with open(klue_dev_path, 'rt', encoding='utf-8') as fIn:
        klue_dev = json.load(fIn)

    for data in klue_train:
        s1 = data['sentence1']
        s2 = data['sentence2']
        score = data['labels']['label'] / 5.0
        train_samples.append(InputExample(texts= [s1,s2], label=score))

    for data in klue_dev:
        s1 = data['sentence1']
        s2 = data['sentence2']
        score = data['labels']['label'] / 5.0
        dev_samples.append(InputExample(texts= [s1,s2], label=score))

    with open(korsts_dev_path, 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines[1:]:
            # print(line)
            _, _, _, _, score, s1, s2 = line.split('\t')
            # print(score, s1, s2)
            score = score.strip()
            score = float(score) / 5.0
            dev_samples.append(InputExample(texts= [s1,s2], label=score))

    with open(korsts_test_path, 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines[1:]:
            # print(line)
            _, _, _, _, score, s1, s2 = line.split('\t')
            # print(score, s1, s2)
            score = score.strip()
            score = float(score) / 5.0
            test_samples.append(InputExample(texts= [s1,s2], label=score))

    with open(korsts_train_path, 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines[1:]:
        # print(line)
            _, _, _, _, score, s1, s2 = line.split('\t')
            # print(score, s1, s2)
            score = score.strip()
            score = float(score) / 5.0
            train_samples.append(InputExample(texts= [s1,s2], label=score))
    return train_samples, dev_samples, test_samples