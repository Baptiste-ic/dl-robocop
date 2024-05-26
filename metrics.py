import os

import matplotlib.pyplot as plt
import nltk
import torch


def compute_metrics(x, y_pred, y_real, judge_model, classifier_tokenizer, threshold, fluency_pipeline,
                    similarity_model, device):
    """
    Compute the BLEU, style accuracy, similarity, fluency and joint metric of the model

    Args:
    - x: The input.
    - y_pred: The predicted output.
    - y_real: The expected output.
    - judge_model: The judge model.
    - classifier_tokenizer: The classifier tokenizer.
    - threshold: The classifier threshold.
    - fluency_pipeline: The fluency pipeline.
    - similarity_model: The similarity model.
    - device: The device.

    Returns:
    - bleu: The BLEU score.
    - sta: The style accuracy.
    - sim: The similarity score.
    - fl: The fluency score.
    - j: The joint metric.
    """

    bleu = bleu_score(y_pred, y_real)
    sta = style_accuracy(y_pred, judge_model, classifier_tokenizer, threshold, device)
    sim = similarity(x, y_pred, similarity_model)
    fl = fluency(y_pred, fluency_pipeline)
    j = sta * sim * fl

    return bleu, sta, sim, fl, j


def bleu_score(y_pred, y_real):
    """
    Compute the BLEU score of the model

    Args:
    - y_pred: The predicted output.
    - y_real: The expected output.

    Returns:
    - bleu: The BLEU score.
    """
    bleu = 0.0
    for i, y_r in enumerate(y_real):
        bleu += nltk.translate.bleu_score.sentence_bleu([y_r], y_pred[i])
    return bleu / len(y_real)


def style_accuracy(y_pred, judge_model, classifier_tokenizer, threshold, device):
    """
    Compute the style accuracy of the model

    Args:
    - y_pred: The predicted output.
    - judge_model: The judge model.
    - classifier_tokenizer: The classifier tokenizer.
    - threshold: The classifier threshold.
    - device: The device.
    """
    acc = 0.0
    for y_p in y_pred:
        judge_input = classifier_tokenizer.encode(y_p, return_tensors='pt').to(device)
        prob = torch.softmax(judge_model(judge_input).logits, dim=-1)[0][1].item()
        acc += int(prob < threshold)
    return acc / len(y_pred)


def similarity(x, y_pred, similarity_model):
    """
    Compute the similarity between the input and the output

    Args:
    - x: The input.
    - y_pred: The predicted output.
    - similarity_model: The similarity model.

    Returns:
    - sim: The similarity score.
    """
    embedding_x, embedding_y_pred = similarity_model.encode([x, y_pred], convert_to_tensor=True)
    return torch.cosine_similarity(embedding_x.unsqueeze(0), embedding_y_pred.unsqueeze(0)).item()


def fluency(y_pred, fluency_pipeline):
    """
    Compute the fluency of the model

    Args:
    - y_pred: The predicted output.
    - fluency_pipeline: The fluency pipeline.

    Returns:
    - fl: The fluency score.
    """
    # Get the classification result
    result = fluency_pipeline(y_pred)

    # Compute the fluency
    fl = 0.0
    for r in result:
        fl += r['label'] == "LABEL_1"
    return fl / len(result)


def plot_metrics(metrics, epoch, metrics_dir, save_frequency):
    """
    Plots the metrics over the epochs.

    Args:
    - metrics: The metrics to plot.
    - epoch: The epoch.
    - metrics_dir: The directory to save the metrics.
    - save_frequency: The frequency to save the metrics.
    """
    # Delete the previous plots
    for file in os.listdir(metrics_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(metrics_dir, file))

    for metric, values in metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2, save_frequency), values, label=f'Epoch vs {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Epoch vs {metric}')
        plt.legend()
        plt.savefig(os.path.join(metrics_dir, f'{metric}_epoch_{epoch + 1}.png'))
        plt.close()

    plt.figure(figsize=(10, 5))
    for metric, values in metrics.items():
        plt.plot(range(1, epoch + 2, save_frequency), values, label=metric)

    plt.xlabel('Epoch')
    plt.ylabel('Metric Score')
    plt.title('Training Metrics Over Epochs')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(metrics_dir, f'combined_metrics_epoch_{epoch + 1}.png'))
    plt.close()


def store_metrics(eval_loss, eval_metrics, epoch, metrics_file_path):
    """
    Stores the metrics in a file.

    Args:
    - eval_loss: The evaluation loss.
    - eval_metrics: The evaluation metrics.
    - epoch: The epoch.
    - metrics_file_path: The path to the metrics file.
    """
    with open(metrics_file_path, 'a') as f:
        f.write(f'Epoch {epoch + 1}: '
                f'Loss {eval_loss:.4f}, '
                f'BLEU {eval_metrics["bleu"]:.4f}, '
                f'STA {eval_metrics["sta"]:.4f}, '
                f'SIM {eval_metrics["sim"]:.4f}, '
                f'FL {eval_metrics["fl"]:.4f}, '
                f'J {eval_metrics["j"]:.4f}\n')
