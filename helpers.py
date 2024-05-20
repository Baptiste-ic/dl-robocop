import torch
import numpy as np
from torch.utils.data import TensorDataset
from bert_score import score
import os
import shutil



def format_data(max_length, tokenizer, dataset):
    '''
    Create a tokenized tensor dataset that can then be used to create dataloaders.
    '''
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


def evaluate(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    tot_loss = 0.0

    # Iterate through the test dataloader
    for batch in test_dataloader:
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

    # Calculate average loss over all batches
    average_loss = tot_loss / len(test_dataloader)

    # Print average loss for the test dataset
    print(f"Test Average Loss: {average_loss:.4f}")
    model.train()


def calculate_toxicity_score(judge_model, classifier_tokenizer, output_sequence, device):
    '''
  Usess a judge model to give a politeness score to the given sequence.
  '''
    # Tokenizing user the judge tokenizer
    # TODO: need to check if we should bring judge_input back to device here or put classifier_tokenizer in device before
    judge_input = classifier_tokenizer.encode(output_sequence, return_tensors='pt').to(device)
    judge_input = judge_input.to(judge_model.device)
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
    '''
  Computes the reinforcement learning loss, it is composed of two parts:

  - First we reward good politenes based on a judge model
  - Then we reward goof Bert scores (f1 score)

  '''
    # Computing politeness scores for the greedy and sampled sentence
    g_toxicity_scores = torch.tensor(
        [calculate_toxicity_score(judge_model, judge_tokenizer, seq, device) for seq in greedy_seq])
    s_toxicity_scores = torch.tensor(
        [calculate_toxicity_score(judge_model, judge_tokenizer, seq, device) for seq in sampled_seq])

    # Rewarding if the sampled sentence is better
    tox_rewards = s_toxicity_scores - g_toxicity_scores

    # Computing BERTScores  for the greedy and sampled sentence
    _, _, g_bert_scores = score(greedy_seq, gold, lang="en", verbose=False)
    _, _, s_bert_scores = score(sampled_seq, gold, lang="en", verbose=False)

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
    '''
    Produces random sequences sampled according to the logits of the model.
    '''

    # Creating variables
    decoder_inputs = {'input_ids': input_ids, 'attention_mask': mask}
    output_ids = torch.tensor([]).to(device)
    sequence_probas = torch.tensor([])
    past_key_values = None

    # Iterating over the sequennces
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


def train_on_paradetox(student_model,
                       judge_model,
                       tokenizer,
                       judge_tokenizer,
                       optimizer,
                       dataloader,
                       test_dataloader,
                       num_epochs,
                       alpha,
                       device,
                       lambda_tox,
                       lambda_bert,
                       weights_dir='weights'):
    ''' 
    Trains a model on the dataset using both Maximum Likelihood and 
    Reinforcement Learning losses.
    '''

    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)
    os.makedirs(weights_dir)

    for epoch in range(num_epochs):
        # Preparing for training
        student_model.train()
        judge_model.eval()
        total_loss = 0.0

        # Iterating over dataset
        for i, batch in enumerate(dataloader):
            print(f'\n {i + 1}/{len(dataloader)}')

            # Getting input, label and mask
            input_ids, detoxified_ids, input_mask = batch
            batch_size = input_ids.size(0)

            # Putting model inputs on device
            input_ids = input_ids.to(device)
            detoxified_ids = detoxified_ids.to(device)
            input_mask = input_mask.to(device)

            # Feeding to student model 
            g_student_outputs = student_model(input_ids=input_ids, attention_mask=input_mask, labels=detoxified_ids)

            # Decoding to greedy sequence to compute RL loss
            g_output_sequences = tokenizer.batch_decode(g_student_outputs.logits.argmax(dim=-1),
                                                        skip_special_tokens=True)

            # Sampling from student model for RL loss
            s_student_outputs, s_student_probas = policy_sampling(student_model, input_ids, input_mask, 40, device)

            # Decodng to sampled sequence to compute RL loss
            s_output_sequences = tokenizer.batch_decode(s_student_outputs, skip_special_tokens=True)

            # Decoding reference sequence for RL loss
            gold_sequences = tokenizer.batch_decode(detoxified_ids, skip_special_tokens=True)

            # Computing Maximum Likelihood loss (usual loss)
            ml_loss = g_student_outputs.loss

            # Computing Reinforcement Learning loss
            rl_loss = compute_rl_loss(judge_model,
                                      judge_tokenizer,
                                      s_student_probas,
                                      g_output_sequences,
                                      s_output_sequences,
                                      gold_sequences,
                                      lambda_tox,
                                      lambda_bert,
                                      device)

            # Computing total loss (RL + ML)
            full_loss = alpha * rl_loss + (1-alpha) * ml_loss

            # Backpropagating
            full_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += full_loss.item()

        # Save model weights
        torch.save(student_model.state_dict(), "./" + weights_dir + '/epoch' + str(epoch) + '.pt')

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average train Loss: {average_loss:.4f}")
        print(f'Evaluating on test set...')
        evaluate(student_model, test_dataloader, device)
        print('\n')
