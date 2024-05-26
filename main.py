import pandas as pd
import torch
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    RobertaTokenizer, RobertaForSequenceClassification, pipeline

from train import format_data, train_on_paradetox

MODEL_NAME = 'facebook/bart-base'
FLUENCY_MODEL_NAME = 'textattack/roberta-base-CoLA'
SIMILARITY_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CLASSIFIER_MODEL_NAME = 'SkolkovoInstitute/roberta_toxicity_classifier'
CLASSIFIER_THRESHOLD = 0.8
DATA_PATH = 'data/paradetox.tsv'
SAVE_DIR = 'weights/'
METRICS_DIR = 'metrics/'
CHECKPOINT_DIR = 'saved_weights/'
CHECKPOINT_MODEL = ''
CHECKPOINT_OPTIMIZER = ''
MAX_TOKENIZATION_LENGTH = 38
TRAIN_SIZE_RATIO = 0.8
VAL_SIZE_RATIO = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 1000
ALPHA_RL_LOSS = 0
LAMBDA_TOX = 0.2
LAMBDA_BERT = 0.8
SAVE_FREQUENCY = 50

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # Setup peft for faster training
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # Main model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

    # Load the weights of the model from the checkpoint if we are resuming training
    if CHECKPOINT_MODEL:
        print('Restoring model weights from checkpoint')
        model.load_state_dict(torch.load(CHECKPOINT_DIR + CHECKPOINT_MODEL, map_location=device))

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Politeness judge
    toxicity_classifier_model = RobertaForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_NAME).to(device)
    toxicity_classifier_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)

    # Fluency pipeline
    # TODO: check that it is indeed the good model (c.f. ParaDetox) and cite it in our paper
    fluency_model = AutoModelForSequenceClassification.from_pretrained(FLUENCY_MODEL_NAME).to(device)
    fluency_tokenizer = AutoTokenizer.from_pretrained(FLUENCY_MODEL_NAME)
    fluency_pipeline = pipeline('text-classification', model=fluency_model, tokenizer=fluency_tokenizer)

    # Similarity model
    # TODO: change model for the one from ParaDetox (p. 8, metrics)
    similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME)

    # Load dataset
    dataset = pd.read_csv(DATA_PATH, sep='\t')[:20]

    # Define the maximum length and other parameters
    tensor_dataset = format_data(MAX_TOKENIZATION_LENGTH, model_tokenizer, dataset)

    # Split the dataset into training, validation and test sets
    test_size_ratio = round(1 - TRAIN_SIZE_RATIO - VAL_SIZE_RATIO, 2)
    train_dataset, val_dataset, test_dataset = random_split(tensor_dataset,
                                                            [TRAIN_SIZE_RATIO, VAL_SIZE_RATIO, test_size_ratio])

    # Defining parameters for training
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Print sizes of the datasets
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Restore optimizer weights if we resume training
    if CHECKPOINT_OPTIMIZER:
        print('Restoring optimizer weights from checkpoint')
        optimizer.load_state_dict(torch.load(CHECKPOINT_DIR + CHECKPOINT_OPTIMIZER, map_location=device))

    # Train
    train_on_paradetox(model=model,
                       tokenizer=model_tokenizer,
                       judge_model=toxicity_classifier_model,
                       judge_tokenizer=toxicity_classifier_tokenizer,
                       judge_threshold=CLASSIFIER_THRESHOLD,
                       fluency_pipeline=fluency_pipeline,
                       similarity_model=similarity_model,
                       optimizer=optimizer,
                       dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       num_epochs=NUM_EPOCHS,
                       alpha=ALPHA_RL_LOSS,
                       device=device,
                       lambda_tox=LAMBDA_TOX,
                       lambda_bert=LAMBDA_BERT,
                       weights_dir=SAVE_DIR,
                       metrics_dir=METRICS_DIR,
                       save_frequency=SAVE_FREQUENCY)
