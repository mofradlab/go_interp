def gibbs_sampler(seq, mask_func, num_iter=16, batch_size=8):
    seq_ind = model.encode(ESMProtein(sequence=seq)).sequence.to(device)
    mask_batch_sample_l = []
    batch_sample_l = [] 
    bert_eval_l = []
    seq_batch = torch.tile(seq_ind.reshape(1, -1), (batch_size, 1))
    with torch.no_grad():
        for _ in range(num_iter):
            update_batch = mask_func(seq_batch)
            batch_sample_l.append(seq_batch.cpu())
            mask_batch_sample_l.append(update_batch.cpu()) #Save masked batch
            model_eval = model(update_batch)
            bert_eval = model_eval.sequence_logits
            bert_eval = torch.softmax(bert_eval[..., 4:24], dim=2)
            N, L, T = bert_eval.shape
            sample_tokens = torch.multinomial(bert_eval.reshape(N*L, T), num_samples=1).reshape(N, L)+4
            update_batch = update_batch * (update_batch != SEQUENCE_MASK_TOKEN) + sample_tokens * (update_batch == SEQUENCE_MASK_TOKEN)
            bert_eval_l.append(bert_eval.cpu())
            seq_batch = update_batch
    return torch.stack(bert_eval_l), torch.stack(batch_sample_l).cpu(), torch.stack(mask_batch_sample_l).cpu()