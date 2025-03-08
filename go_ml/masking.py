import torch

def mask_range(seq_batch, si, ei, sequence_mask_token, mut_per=0.15):
    device = seq_batch.device
    batch_size, seq_len = seq_batch.shape
    # seq_len = torch.LongTensor([seq_len - 2]) #Discount SOS and EOS tokens at start and end
    range_len = torch.LongTensor([ei-si])
    mut_count = torch.floor(range_len*mut_per).int().item()
    mut_inds = torch.stack([torch.randperm(range_len) for _ in range(batch_size)])[:, :mut_count] + si
    batch_inds = torch.tile(torch.arange(0, mut_inds.shape[0]).reshape((-1, 1)), (1, mut_count))
    mut_inds, batch_inds = mut_inds.to(device), batch_inds.to(device)
    update_batch = seq_batch.clone()
    update_batch[batch_inds, mut_inds] = sequence_mask_token
    return update_batch

#Masking functions for ESM3 only. Requires input to be augmented with SOS and EOS tokens
def mask_perc(seq_ind, mask_token, residue_coverage=6, mut_per=0.15):
    device = seq_ind.device
    seq_len = torch.LongTensor([seq_ind.shape[0] - 2]).to(device) #Discount SOS and EOS tokens at start and end
    mut_count = torch.floor(seq_len*mut_per).int().item()
    total_muts = (torch.floor(seq_len*residue_coverage/mut_count)*mut_count).int().item()
    
    mut_inds = (torch.randperm(total_muts).reshape(-1, mut_count).to(device) % seq_len) + 1
    batch_inds = torch.tile(torch.arange(0, mut_inds.shape[0]).reshape((-1, 1)), (1, mut_count))
    mut_inds, batch_inds = mut_inds.to(device), batch_inds.to(device)

    batch = torch.tile(seq_ind, (mut_inds.shape[0], 1))
    batch[batch_inds, mut_inds] = mask_token
    # print((batch==mask_token).sum())
    return batch, batch_inds, mut_inds

def mask_indiv(seq_ind, mask_token):
    seq_len = seq_ind.shape[0] - 2 #Discount first and last tokens
    batch = torch.tile(seq_ind, (seq_len, 1))
    batch_ind = torch.arange(seq_len) 
    mut_ind = batch_ind + 1
    batch[batch_ind, mut_ind] = mask_token
    return batch, batch_ind, mut_ind

def mask_indiv_neighborhood(seq_ind, mask_token, n_rad=5):
    seq_len = seq_ind.shape[0] - 2 #Discount first and last tokens
    batch = torch.tile(seq_ind, (seq_len, 1))
    batch_ind = torch.arange(seq_len)
    mut_ind = batch_ind + 1
    col_ind = torch.arange(seq_ind.shape[0]).reshape(1, -1)
    col_ind = torch.tile(col_ind, (seq_len, 1))
    mut_delta = mut_ind.reshape(-1, 1) - col_ind
    mut_mask = torch.abs(mut_delta) <= n_rad
    mut_mask[:, 0] = False; mut_mask[:, -1] = False #Don't mess with sos, eos tokens
    batch[mut_mask] = mask_token
    return batch, batch_ind, mut_ind

def mask_avg(bert_mask, bert_eval):
    eval_idx = torch.nonzero(bert_mask) ##Nonzero x 2(ij)
    eval_support = torch.sum(bert_mask, dim=0)
    eval_samples = bert_eval[eval_idx[:, 0], eval_idx[:, 1], :]
    eval_avg = torch.zeros_like(bert_eval[0])
    gather_ind = torch.tile(eval_idx[:, 1:2], (1, bert_eval.shape[-1]))
    eval_avg = eval_avg.scatter_reduce(0, gather_ind, eval_samples, 'mean', include_self=False)
    return eval_avg, eval_support