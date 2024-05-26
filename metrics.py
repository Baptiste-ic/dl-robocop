import torch
import nltk


# TODO: cite https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 in our paper

def compute_metrics(x, y_pred, y_real, judge_model, classifier_tokenizer, threshold, fluency_pipeline,
                    similarity_model, device):
    """
    Compute the BLEU, style accuracy, similarity, fluency and joint metric of the model
    """

    bleu = 0.0
    for i, y_r in enumerate(y_real):
        bleu += nltk.translate.bleu_score.sentence_bleu([y_r], y_pred[i])
    bleu /= len(y_real)
    sta = style_accuracy(y_pred, judge_model, classifier_tokenizer, threshold, device)
    sim = similarity(x, y_pred, similarity_model)
    fl = fluency(y_pred, fluency_pipeline)
    j = sta * sim * fl

    return bleu, sta, sim, fl, j


def style_accuracy(y_pred, judge_model, classifier_tokenizer, threshold, device):
    acc = 0.0
    for y_p in y_pred:
        judge_input = classifier_tokenizer.encode(y_p, return_tensors='pt').to(device)
        prob = torch.softmax(judge_model(judge_input).logits, dim=-1)[0][1].item()
        acc += int(prob < threshold)
    return acc / len(y_pred)


def similarity(x, y_pred, similarity_model):
    embeding_x, embeding_y_pred = similarity_model.encode([x, y_pred], convert_to_tensor=True)
    return torch.cosine_similarity(embeding_x.unsqueeze(0), embeding_y_pred.unsqueeze(0)).item()


def fluency(y_pred, fluency_pipeline):
    # Get the classification result
    result = fluency_pipeline(y_pred)

    # Compute the fluency
    fl = 0.0
    for r in result:
        fl += r['label'] == "LABEL_1"
    return fl / len(result)