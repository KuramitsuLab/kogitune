import torch

def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
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
    return torch.exp(loss).item(), all_prob, loss.item()


def inference_one_model(model1, tokenizer1, text, ex, modelname1):
    pred = {}

    # model1
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

   # ppl
    pred["ppl"] = p1
    # min-k prob
    # for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred
    return ex


def evaluate_data_one_model(test_data, model1, tokenizer1, col_name, modelname1):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data):
        text = ex[col_name]
        # new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2)
        new_ex = inference_one_model(model1, tokenizer1, text, ex, modelname1)
        all_output.append(new_ex)
    return all_output