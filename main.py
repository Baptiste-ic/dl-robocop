import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel
from torch.utils.data.dataset import random_split
from helpers import format_data
from helpers import train_on_paradetox

DATA_PATH = "data/"
MAX_TOKENIZATION_LENGTH = 38
TRAIN_SIZE_RATIO = 0.8
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 1000
ALPHA_RL_LOSS = 0.5
LAMBDA_TOX = 0.2
LAMBDA_BERT = 0.8

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # Ensure task type aligns with sequence-to-sequence learning
        inference_mode=False,  # Set to False for training, True for inference
        r=8,  # The rank of the LoRA matrices
        lora_alpha=32,  # The scaling factor for the LoRA matrices
        lora_dropout=0.1  # Dropout rate to use in LoRA layers
    )

    # BART MODEL (STUDENT)
    model_name = "facebook/bart-base"
    student_tokenizer = AutoTokenizer.from_pretrained(model_name)
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    student_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    student_model = get_peft_model(student_model, peft_config)
    student_model.print_trainable_parameters()

    # student_model = AutoModel.from_pretrained(model_name).to(device)

    # POLITENESS JUDGE

    # load tokenizer and model weights
    # classifier_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    # toxicity_clasifier_model = RobertaForSequenceClassification.from_pretrained(
    #    'SkolkovoInstitute/roberta_toxicity_classifier')

    classifier_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    toxicity_classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_name).to(device)

    weights_path = "model.pth"
    student_model.load_state_dict(torch.load(weights_path, map_location=device))

    data_path = DATA_PATH
    dataset = pd.read_csv(data_path + 'paradetox.tsv', sep='\t')

    # Define the maximum length and other parameters
    max_length = MAX_TOKENIZATION_LENGTH
    tensor_dataset = format_data(max_length, student_tokenizer, dataset)

    # Define the size of the training set
    train_size = int(TRAIN_SIZE_RATIO * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size

    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])

    # Defining parameters for training
    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    num_epochs = NUM_EPOCHS

    train_on_paradetox(student_model=student_model,
                       judge_model=toxicity_classifier_model,
                       tokenizer=student_tokenizer,
                       judge_tokenizer=classifier_tokenizer,
                       optimizer=optimizer,
                       dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       num_epochs=num_epochs,
                       alpha=ALPHA_RL_LOSS,
                       device=device,
                       lambda_tox=LAMBDA_TOX,
                       lambda_bert=LAMBDA_BERT,
                       weights_dir="weights_Auto")
