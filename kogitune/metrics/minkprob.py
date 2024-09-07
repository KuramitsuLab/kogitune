import torch
import numpy as np

def calc_prob(model, tokenizer, device, record):
    """
    exp(loss)
    """
    sentence = record['input']
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    # return torch.exp(loss).item(), all_prob, loss.item()
    # p1 = torch.exp(loss).item()
    # all_prob
    record['loss'] = loss.item()
    record['ppl'] = torch.exp(loss).item()
    pred={}
    sorted_prob = np.sort(all_prob)
    for top_k in range(5, 101, 5):
        topk_prob = sorted_prob[:top_k]
        pred[f"min{top_k}_prob"] = -np.mean(topk_prob).item()
    record['mink_prob'] = pred
    return record
