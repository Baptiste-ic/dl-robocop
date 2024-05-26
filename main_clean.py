import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, \
    RobertaTokenizer, RobertaForSequenceClassification, pipeline
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data.dataset import random_split
from helpers import format_data
from helpers import train_on_paradetox
from sentence_transformers import SentenceTransformer


MODEL_NAME = 'facebook/bart-base'
CLASSIFIER_NAME = 'SkolkovoInstitute/roberta_toxicity_classifier'  # or "facebook/roberta-hate-speech-dynabench-r4-target"
CLASSIFIER_THRESHOLD = 0.8
DATA_PATH = 'data/paradetox.tsv'
SAVE_DIR = 'weights/'
CHECKPOINT_DIR = 'saved_weights/'
CHECKPOINT_MODEL = 'epoch100_no_rl_loss.pth'  # model.pth
# We dont use a scheduler for now
# CHECKPOINT_SCHEDULER = ''
CHECKPOINT_OPTIMIZER = ''
MAX_TOKENIZATION_LENGTH = 38
TRAIN_SIZE_RATIO = 0.8
VAL_SIZE_RATIO = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 1000
ALPHA_RL_LOSS = 0.5
LAMBDA_TOX = 0.2
LAMBDA_BERT = 0.8

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # BART MODEL (STUDENT)
    model_name = "facebook/bart-base"
    student_tokenizer = AutoTokenizer.from_pretrained(model_name)
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    student_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if CHECKPOINT_MODEL:
        print("Restoring model weights from checkpoint")
        student_model.load_state_dict(torch.load(CHECKPOINT_DIR + CHECKPOINT_MODEL, map_location=device))

    student_model = get_peft_model(student_model, peft_config)
    student_model.print_trainable_parameters()

    # POLITENESS JUDGE
    classifier_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_NAME)
    toxicity_classifier_model = RobertaForSequenceClassification.from_pretrained(CLASSIFIER_NAME).to(device)

    # Fluency pipeline
    fluency_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    # TODO: check that it is indeed the good model (c.f. ParaDetox) and cite it in our paper
    fluency_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(device)
    fluency_pipeline = pipeline("text-classification", model=fluency_model, tokenizer=fluency_tokenizer)

    # Similarity model
    # TODO: change model for the one from ParaDetox (p. 8, metrics)
    similarity_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Load dataset
    dataset = pd.read_csv(DATA_PATH, sep='\t')[:30]

    # Define the maximum length and other parameters
    tensor_dataset = format_data(MAX_TOKENIZATION_LENGTH, student_tokenizer, dataset)

    # Split the dataset into training, validation and test sets
    test_size_ratio = round(1 - TRAIN_SIZE_RATIO - VAL_SIZE_RATIO, 2)
    train_dataset, val_dataset, test_dataset = random_split(tensor_dataset,
                                                            [TRAIN_SIZE_RATIO, VAL_SIZE_RATIO, test_size_ratio])

    # Defining parameters for training
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Print sizes of the datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
    if CHECKPOINT_OPTIMIZER:
        print("Restoring optimizer weights from checkpoint")
        optimizer.load_state_dict(torch.load(CHECKPOINT_DIR + CHECKPOINT_OPTIMIZER, map_location=device))

    train_on_paradetox(student_model=student_model,
                       judge_model=toxicity_classifier_model,
                       judge_threshold=CLASSIFIER_THRESHOLD,
                       fluency_pipeline=fluency_pipeline,
                       similarity_model=similarity_model,
                       tokenizer=student_tokenizer,
                       judge_tokenizer=classifier_tokenizer,
                       optimizer=optimizer,
                       dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       num_epochs=NUM_EPOCHS,
                       alpha=ALPHA_RL_LOSS,
                       device=device,
                       lambda_tox=LAMBDA_TOX,
                       lambda_bert=LAMBDA_BERT,
                       weights_dir="weights/")

    # TODO: test final model
    # test(test_dataloader ...)
