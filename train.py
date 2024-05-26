import os
from datetime import datetime

import torch
from bert_score import score
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset

from metrics import compute_metrics, store_metrics, plot_metrics


def format_data(max_length: int, tokenizer, dataset):
    """
    Create a tokenized tensor dataset that can then be used to create dataloaders.

    Args:
    - max_length: The maximum length of the tokenized sequences.
    - tokenizer: The tokenizer to use.
    - dataset: The dataset to format.

    Returns:
    - tensor_dataset: The tokenized tensor dataset.
    """
    # Generate the input_ids and input_mask for each sample
    input_ids = [torch.tensor(tokenizer.encode(d, max_length=max_length, padding='max_length')) for d in
                 dataset['toxic']]
    detoxified_ids = [torch.tensor(tokenizer.encode(d, max_length=max_length, padding='max_length')) for d in
                      dataset['neutral1']]
    input_mask = [torch.tensor([1 if token != tokenizer.pad_token_id else 0 for token in tokens]) for tokens in
                  input_ids]

    # Convert lists to tensors
    input_ids_tensor = torch.stack(input_ids)
    detoxified_ids_tensor = torch.stack(detoxified_ids)
    input_mask_tensor = torch.stack(input_mask)

    # Create a TensorDataset
    tensor_dataset = TensorDataset(input_ids_tensor, detoxified_ids_tensor, input_mask_tensor)

    return tensor_dataset


def evaluate(model, tokenizer, judge_model, judge_tokenizer, judge_threshold, val_dataloader, fluency_pipeline,
             similarity_model, device):
    """
    Evaluate the model on the validation set.

    Args:
    - model: The model to evaluate.
    - tokenizer: The tokenizer to use.
    - judge_model: The model to use as a judge for politeness.
    - judge_tokenizer: The tokenizer to use for the judge model.
    - judge_threshold: The threshold of the judge.
    - val_dataloader: The dataloader to use for validation.
    - fluency_pipeline: The pipeline to compute the fluency.
    - similarity_model: The model to compute the similarities.
    - device: The device to use.

    Returns:
    - average_loss: The average loss on the validation set.
    - metrics: The metrics on the validation set.
    """
    # Set the model to evaluation mode
    model.eval()
    tot_loss = 0.0

    metrics = {'bleu': 0,
               'sta': 0,
               'sim': 0,
               'fl': 0,
               'j': 0}

    # Iterate through the test dataloader
    for batch in val_dataloader:
        input_ids, detoxified_ids, input_mask = batch

        # Move tensors to the device
        input_ids = input_ids.to(device)
        detoxified_ids = detoxified_ids.to(device)
        input_mask = input_mask.to(device)

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            model_outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=detoxified_ids)
            loss = model_outputs.loss

            # Accumulate total loss
            tot_loss += loss.item()

            # Compute metrics
            x = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            y_pred = tokenizer.batch_decode(model_outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            y_real = tokenizer.batch_decode(detoxified_ids, skip_special_tokens=True)
            bleu, sta, sim, fl, j = compute_metrics(x, y_pred, y_real, judge_model, judge_tokenizer, judge_threshold,
                                                    fluency_pipeline, similarity_model, device)
            metrics['bleu'] += bleu
            metrics['sta'] += sta
            metrics['sim'] += sim
            metrics['fl'] += fl
            metrics['j'] += j

    # Calculate average loss and metrics over all batches
    average_loss = tot_loss / len(val_dataloader)
    for key in metrics:
        metrics[key] /= len(val_dataloader)

    # Print average loss and metrics for the validation set
    print(f'Test Average Loss: {average_loss:.4f}')
    print(f'Test BLEU: {metrics["bleu"]:.4f}')
    print(f'Test STA: {metrics["sta"]:.4f}')
    print(f'Test SIM: {metrics["sim"]:.4f}')
    print(f'Test FL: {metrics["fl"]:.4f}')
    print(f'Test J: {metrics["j"]:.4f}')

    model.train()
    return average_loss, metrics


def calculate_toxicity_score(judge_model, classifier_tokenizer, output_sequence, device):
    """
    Uses a judge model to give a politeness score to the given sequence.

    Args:
    - judge_model: The model to use as a judge for politeness.
    - classifier_tokenizer: The tokenizer to use for the judge model.
    - output_sequence: The sequence to evaluate.
    - device: The device to use.

    Returns:
    - toxicity_score: The politeness score.
    """
    # Tokenizing user the judge tokenizer
    judge_input = classifier_tokenizer.encode(output_sequence, return_tensors='pt').to(device)
    # NOTE: This is a toy method because I did not find any pretrained politeness classifier!
    politeness_distribution = torch.softmax(judge_model(judge_input).logits, dim=-1)[0]
    toxicity_indicator = torch.argmax(politeness_distribution).item()
    toxicity_score = (((-1) ** toxicity_indicator * politeness_distribution[toxicity_indicator]) + 1) / 2
    return toxicity_score


def compute_rl_loss(judge_model,
                    judge_tokenizer,
                    sampled_probas,
                    greedy_seq,
                    sampled_seq,
                    gold,
                    lambda_tox,
                    lambda_bert,
                    device):
    """
     Computes the reinforcement learning loss, it is composed of two parts:

    - First we reward good politeness based on a judge model
    - Then we reward goof Bert scores (f1 score)

    Args:
    - judge_model: The model to use as a judge for politeness.
    - judge_tokenizer: The tokenizer to use for the judge model.
    - sampled_probas: The probabilities of the sampled sequences.
    - greedy_seq: The greedy sequence.
    - sampled_seq: The sampled sequence.
    - gold: The gold sequence.
    - lambda_tox: The weight to give to the politeness score.
    - lambda_bert: The weight to give to the BERT score.
    - device: The device to use.

    Returns:
    - rl_loss: The reinforcement learning loss.

    """
    # Computing politeness scores for the greedy and sampled sentence
    g_toxicity_scores = torch.tensor(
        [calculate_toxicity_score(judge_model, judge_tokenizer, seq, device) for seq in greedy_seq])
    s_toxicity_scores = torch.tensor(
        [calculate_toxicity_score(judge_model, judge_tokenizer, seq, device) for seq in sampled_seq])

    # Rewarding if the sampled sentence is better
    tox_rewards = s_toxicity_scores - g_toxicity_scores

    # Computing BERTScores  for the greedy and sampled sentence
    _, _, g_bert_scores = score(greedy_seq, gold, lang='en', verbose=False)
    _, _, s_bert_scores = score(sampled_seq, gold, lang='en', verbose=False)

    # Rewarding if the sampled sentence is better
    bert_rewards = s_bert_scores - g_bert_scores

    # Computing total reward with respective weights
    total_rewards = lambda_tox * tox_rewards + lambda_bert * bert_rewards

    # Loss is the negative log likelihood of the sampled sentence
    sum_probas = torch.sum(torch.log(sampled_probas), dim=-1)

    # Computing loss with rewards
    rl_loss = torch.mean(total_rewards * sum_probas)
    return rl_loss


def policy_sampling(model, input_ids, mask, max_seq_length, device):
    """
    Produces random sequences sampled according to the logits of the model.

    Args:
    - model: The model to use for sampling.
    - input_ids: The input ids to use.
    - mask: The mask to use.
    - max_seq_length: The maximum length of the sequences to generate.
    - device: The device to use.

    Returns:
    - output_ids: The sampled output ids.
    - sequence_probas: The probabilities of the sampled sequences.
    """

    # Creating variables
    decoder_inputs = {'input_ids': input_ids, 'attention_mask': mask}
    output_ids = torch.tensor([]).to(device)
    sequence_probas = torch.tensor([])
    past_key_values = None

    # Iterating over the sequences
    for t in range(max_seq_length):
        # Feeding last generation
        model_outputs = model(**decoder_inputs, use_cache=True, past_key_values=past_key_values)

        # Storing what the model has already computed (for speedup)
        past_key_values = model_outputs['past_key_values']

        # Getting logits
        logits = model_outputs.logits[:, -1, :]

        # Computing true distribution
        probabilities = torch.softmax(logits, dim=-1)

        # Sample from the modified distribution
        next_tokens = torch.multinomial(probabilities, 1).to(device)

        # Getting probabilities corresponding to sampled tokens
        probabilities = torch.gather(input=probabilities, dim=1, index=next_tokens).to('cpu')

        # Setting input as the freshly generated tokens (rest of the sequence in past_key_values)
        decoder_inputs['input_ids'] = next_tokens

        # Creating new mask
        decoder_inputs['attention_mask'] = torch.ones_like(next_tokens).unsqueeze(1)

        # Concatenating generation to output
        output_ids = torch.cat([output_ids, next_tokens], dim=1)

        # Concatenation probabilities of generated tokens
        sequence_probas = torch.cat([sequence_probas, probabilities], dim=1)

    # Converting to int
    output_ids = output_ids.to(torch.int)

    return output_ids, sequence_probas


def sample_sequence(logits):
    """
    Samples a sequence from the logits.

    Args:
    - logits: The logits to sample from.

    Returns:
    - sampled_indices: The sampled indices.
    - probabilities: The probabilities of the sampled indices.
    """
    # Compute probabilities using softmax
    probabilities = softmax(logits, dim=-1).cpu()
    sampled_indices = torch.zeros((logits.size(0), logits.size(1), 1), dtype=torch.long)

    # Sample tokens using multinomial
    for i in range(logits.shape[1]):
        sampled_indices[:, i, :] = torch.multinomial(probabilities[:, i, :], 1)

    probabilities = torch.gather(probabilities, 2, sampled_indices).squeeze()
    sampled_indices = sampled_indices.squeeze()
    return sampled_indices, probabilities


def train_on_paradetox(model,
                       judge_model,
                       tokenizer,
                       judge_tokenizer,
                       judge_threshold,
                       fluency_pipeline,
                       similarity_model,
                       optimizer,
                       dataloader,
                       val_dataloader,
                       num_epochs,
                       alpha,
                       device,
                       lambda_tox,
                       lambda_bert,
                       weights_dir,
                       metrics_dir,
                       save_frequency):
    """
    Trains a model on the dataset using both Maximum Likelihood and
    Reinforcement Learning losses.

    Args:
    - model: The model to train.
    - judge_model: The model to use as a judge for politeness.
    - tokenizer: The tokenizer of the main model.
    - judge_tokenizer: The tokenizer to use for the judge model.
    - judge_threshold: The threshold of the judge.
    - fluency_pipeline: The pipeline to compute the fluency
    - similarity_model: The model to compute the similarities
    - optimizer: The optimizer to use.
    - dataloader: The dataloader to use for training.
    - val_dataloader: The dataloader to use for validation.
    - num_epochs: The number of epochs to train for.
    - alpha: The weight to give to the RL loss.
    - device: The device to use for training.
    - lambda_tox: The weight to give to the politeness score.
    - lambda_bert: The weight to give to the BERT score.
    - weights_dir: The directory to save the model weights.
    - metrics_dir: The directory to save the metrics.
    - save_frequency: The frequency to save the weights
    """

    # Create new directories for weights
    unique_name = str(datetime.now())[:-7].replace(' ', ',').replace(':', '-') + '/'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    weights_dir += unique_name
    os.makedirs(weights_dir)

    # Create new directories for metrics
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    metrics_dir += unique_name
    os.makedirs(metrics_dir)
    metrics_file_path = os.path.join(metrics_dir, 'metrics.txt')

    metrics = {'loss': [],
               'bleu': [],
               'sta': [],
               'sim': [],
               'fl': [],
               'j': []}

    for epoch in range(num_epochs):
        # Preparing for training
        model.train()
        judge_model.eval()
        total_loss = 0.0

        # Iterating over dataset
        for i, batch in enumerate(dataloader):
            print(f'\r {i + 1}/{len(dataloader)}', end='')

            # Getting input, label and mask
            input_ids, detoxified_ids, input_mask = batch

            # Putting model inputs on device
            input_ids = input_ids.to(device)
            detoxified_ids = detoxified_ids.to(device)
            input_mask = input_mask.to(device)

            # Feeding to main model
            g_model_outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=detoxified_ids)

            # Computing Maximum Likelihood loss (usual loss)
            ml_loss = g_model_outputs.loss
            full_loss = ml_loss
            if alpha != 0:
                # Decoding to greedy sequence to compute RL loss
                g_output_sequences = tokenizer.batch_decode(g_model_outputs.logits.argmax(dim=-1),
                                                            skip_special_tokens=True)

                # Sampling from main model for RL loss
                s_model_outputs, s_model_probas = sample_sequence(g_model_outputs.logits)

                # Decoding to sampled sequence to compute RL loss
                s_output_sequences = tokenizer.batch_decode(s_model_outputs, skip_special_tokens=True)

                # Decoding reference sequence for RL loss
                gold_sequences = tokenizer.batch_decode(detoxified_ids, skip_special_tokens=True)

                # Computing Reinforcement Learning loss
                rl_loss = compute_rl_loss(judge_model,
                                          judge_tokenizer,
                                          s_model_probas,
                                          g_output_sequences,
                                          s_output_sequences,
                                          gold_sequences,
                                          lambda_tox,
                                          lambda_bert,
                                          device)

                # Computing total loss (RL + ML)
                full_loss += alpha * rl_loss

            # Backpropagating
            full_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += full_loss.item()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Average train Loss: {average_loss:.4f}')

        if epoch % save_frequency == 0:
            print('Saving model weights and optimizer state...')
            # Save model weights
            torch.save(model.state_dict(), './' + weights_dir + 'model_epoch' + str(epoch) + '.pt')
            # Save optimizer state
            torch.save(optimizer.state_dict(), './' + weights_dir + 'optimizer_epoch' + str(epoch) + '.pt')

            print(f'Evaluating on test set...')
            eval_loss, eval_metrics = evaluate(model, tokenizer, judge_model, judge_tokenizer, judge_threshold,
                                               val_dataloader, fluency_pipeline, similarity_model, device)
            # Store eval metrics
            metrics['loss'].append(eval_loss)
            metrics['bleu'].append(eval_metrics['bleu'])
            metrics['sta'].append(eval_metrics['sta'])
            metrics['sim'].append(eval_metrics['sim'])
            metrics['fl'].append(eval_metrics['fl'])
            metrics['j'].append(eval_metrics['j'])
            store_metrics(eval_loss, eval_metrics, epoch, metrics_file_path)

            # Plot the metrics
            plot_metrics(metrics, epoch, metrics_dir, save_frequency)
            print('\n')
