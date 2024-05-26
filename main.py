import argparse
import os
import warnings

import pandas as pd
import torch
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    RobertaTokenizer, RobertaForSequenceClassification, pipeline, logging

from train import format_data, train_on_paradetox

# Parse program arguments
parser = argparse.ArgumentParser(description="Script to run a model with specified parameters.")

parser.add_argument('--inference', action='store_true', default=False,
                    help='Add this arguent to perform inference instead of training')
parser.add_argument('--model_name', type=str, default='facebook/bart-base', help='Name of the main model')
parser.add_argument('--fluency_model_name', type=str, default='textattack/roberta-base-CoLA',
                    help='Name of the model to compute fluency')
parser.add_argument('--similarity_model_name', type=str,
                    default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    help='Name of the similarity model')
parser.add_argument('--classifier_model_name', type=str,
                    default='facebook/roberta-hate-speech-dynabench-r4-target',
                    help='Name of the classifier model')
parser.add_argument('--classifier_threshold', type=float, default=0.8, help='Threshold for the classifier')
parser.add_argument('--data_path', type=str, default='data/paradetox.tsv', help='Path to the data file')
parser.add_argument('--save_dir', type=str, default='weights', help='Directory to save weights')
parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory to save metrics')
parser.add_argument('--checkpoint_dir', type=str, default='saved_weights', help='Directory to saved checkpoints')
parser.add_argument('--checkpoint_model', type=str, default='', help='Path to checkpoint model')
parser.add_argument('--checkpoint_optimizer', type=str, default='', help='Path to checkpoint optimizer')
parser.add_argument('--max_tokenization_length', type=int, default=38, help='Maximum tokenization length')
parser.add_argument('--train_size_ratio', type=float, default=0.8, help='Ratio of training data size')
parser.add_argument('--val_size_ratio', type=float, default=0.1, help='Ratio of validation data size')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--warmup_epochs', type=int, default=100,
                    help='Number of warmup epochs (during which we don\'t use the rl loss')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--alpha_rl_loss', type=float, default=0, help='Alpha RL loss')
parser.add_argument('--lambda_tox', type=float, default=0.2, help='Lambda toxicity')
parser.add_argument('--lambda_bert', type=float, default=0.8, help='Lambda BERT')
parser.add_argument('--save_frequency', type=int, default=50, help='Frequency of saving weights')


def main():
    args = parser.parse_args()

    # Remove irrelevant warnings
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

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

    if args.inference and not args.checkpoint_model:
        raise ValueError('You must provide a checkpoint model to perform inference')

    # Main model
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

    # Load the weights of the model from the checkpoint if we are resuming training
    if args.checkpoint_model:
        print('Restoring model weights from checkpoint')
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_model), map_location=device))

    if args.inference:
        inference(model, model_tokenizer, args.max_tokenization_length)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Politeness judge
    toxicity_classifier_model = RobertaForSequenceClassification.from_pretrained(args.classifier_model_name).to(device)
    toxicity_classifier_tokenizer = RobertaTokenizer.from_pretrained(args.classifier_model_name)

    # Fluency pipeline
    fluency_model = AutoModelForSequenceClassification.from_pretrained(args.fluency_model_name).to(device)
    fluency_tokenizer = AutoTokenizer.from_pretrained(args.fluency_model_name)
    fluency_pipeline = pipeline('text-classification', model=fluency_model, tokenizer=fluency_tokenizer)

    # Similarity model
    similarity_model = SentenceTransformer(args.similarity_model_name)

    # Load dataset
    dataset = pd.read_csv(args.data_path, sep='\t')

    # Define the maximum length and other parameters
    tensor_dataset = format_data(args.max_tokenization_length, model_tokenizer, dataset)

    # Split the dataset into training, validation and test sets
    test_size_ratio = round(1 - args.train_size_ratio - args.val_size_ratio, 2)
    train_dataset, val_dataset, test_dataset = random_split(tensor_dataset,
                                                            [args.train_size_ratio, args.val_size_ratio,
                                                             test_size_ratio])

    # Defining parameters for training
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Print sizes of the datasets
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Restore optimizer weights if we resume training
    if args.checkpoint_optimizer:
        print('Restoring optimizer weights from checkpoint')
        optimizer.load_state_dict(
            torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_optimizer), map_location=device))

    # Train
    train_on_paradetox(model=model,
                       tokenizer=model_tokenizer,
                       judge_model=toxicity_classifier_model,
                       judge_tokenizer=toxicity_classifier_tokenizer,
                       judge_threshold=args.classifier_threshold,
                       fluency_pipeline=fluency_pipeline,
                       similarity_model=similarity_model,
                       optimizer=optimizer,
                       dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       num_epochs=args.num_epochs,
                       warmup_epochs=args.warmup_epochs,
                       alpha=args.alpha_rl_loss,
                       device=device,
                       lambda_tox=args.lambda_tox,
                       lambda_bert=args.lambda_bert,
                       weights_dir=args.save_dir,
                       metrics_dir=args.metrics_dir,
                       save_frequency=args.save_frequency)


def inference(model, tokenizer, max_length):
    while True:
        sentence = input("Please enter a sentence to detoxify [enter empty string to quit]: ")
        if not sentence:
            print("Terminating...")
            exit(0)
        input_ids = torch.tensor(tokenizer.encode(sentence, max_length=max_length, padding='max_length'))
        input_mask = torch.tensor([1 if token != tokenizer.pad_token_id else 0 for token in input_ids])

        with torch.no_grad():
            output = model(input_ids.unsqueeze(0).to(model.device),
                           attention_mask=input_mask.unsqueeze(0).to(model.device))
            output_sentence = tokenizer.batch_decode(output.logits.argmax(dim=-1), skip_special_tokens=True)
            print(output_sentence)


if __name__ == '__main__':
    main()
