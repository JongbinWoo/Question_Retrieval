from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import logging
from datetime import datetime

from utils import load_parapharse_dataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
def main():

    model_name = "kykim/bert-kor-base"
    train_batch_size = 16
    num_epochs = 4
    model_save_path = 'output/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = CrossEncoder(model_name, num_labels=1)

    logging.info("Read STS dataset")

    train_samples, dev_samples = load_parapharse_dataset()

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    model.fit(train_dataloader=train_dataloader, 
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path)


if __name__ == '__main__':
    main()