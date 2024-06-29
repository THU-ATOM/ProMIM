import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import AUROC

class PIM(nn.Module):
    def __init__(self, patch_size=128, temperature=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.temperature = temperature


    def contrastive_loss(self, similarity_matrix, batch_size):

        input_similarity_matrix = similarity_matrix.clone()
        softmax_row = F.softmax(input_similarity_matrix / self.temperature, dim=1)
        loss_row = (- 1 / batch_size * torch.log(torch.diag(softmax_row))).sum()
        softmax_column = F.softmax(input_similarity_matrix / self.temperature, dim=0)
        loss_column = (- 1 / batch_size * torch.log(torch.diag(softmax_column))).sum()
        total_contrastive_loss = 0.5 * (loss_row + loss_column)

        return total_contrastive_loss


    def forward(self, repre):

        batch_size = repre.shape[0]

        # Split
        repre_1 = repre[:,:self.patch_size,:]
        repre_2 = repre[:,self.patch_size:,:]

        # Max Pooling
        M_repre_1 = repre_1.max(dim=1)[0]
        M_repre_2 = repre_2.max(dim=1)[0]

        # Similarity Matrix
        M_repre_1 = M_repre_1.unsqueeze(1)
        M_repre_2 = M_repre_2.unsqueeze(0)
        similarity_matrix = F.cosine_similarity(M_repre_1, M_repre_2, dim=-1)        

        # Contrastive Loss
        total_contrastive_loss = self.contrastive_loss(similarity_matrix, batch_size)

        # AUC
        with torch.no_grad():
            flat_similarity_matrix = similarity_matrix.clone().view(-1)
            auc_labels = torch.zeros(batch_size * batch_size).to(repre)
            auc_labels[torch.arange(batch_size) * (batch_size + 1)] = 1
            auroc = AUROC(task="binary")
            contra_auc = torch.tensor([auroc(flat_similarity_matrix, auc_labels).item()]).to(repre)

        # ACC
        def accuracy(output, target, topk):       
            with torch.no_grad():
                maxk = max(topk)
                size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)

                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(1.0 / size))
                return res

        acc_labels = torch.tensor(list(range(batch_size))).to(repre)
        contra_top1_acc, contra_top5_acc = accuracy(similarity_matrix, acc_labels, topk=(1, 5))

        return total_contrastive_loss, contra_auc, contra_top1_acc, contra_top5_acc 
