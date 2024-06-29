import torch
import torch.nn as nn

from src.modules.encoders.single import PerResidueEncoder
from src.modules.encoders.pair_wo_dist_dihed import ResiduePairEncoder
from src.modules.encoders.attn import GAEncoder
from src.utils.protein.constants import BBHeavyAtom
from src.modules.pim import PIM
from src.modules.bim import BIM


class ProMIM(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.encoder.node_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )

        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB
        )

        self.attn_encoder = GAEncoder(**cfg.encoder)

        # Protein-level Interaction Modeling
        self.pim = PIM(patch_size=64)

        # Backbone-level Interaction Modeling
        self.bim = BIM(cfg.tf_encoder)


    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi_native']

        x = self.single_encoder(
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,
        )

        z = self.pair_encoder(
            aa = batch['aa'], 
            res_nb = batch['res_nb'], chain_nb = batch['chain_nb'],
            mask_atoms = batch['mask_atoms']
        )

        x = self.attn_encoder(
            pos_atoms = batch['transformed_pos_atoms'],
            res_feat = x, pair_feat = z, 
            mask = mask_residue
        )

        return x


    def forward(self, batch):

        chi_native = torch.where(
            torch.rand_like(batch['chi']) > 0,
            batch['chi'],
            batch['chi_alt']
        )
        batch['chi_native'] = chi_native

        batch_repre = self.encode(batch)

        if self.cfg.train_mode == 'pim+bim':
            contrastive_loss, contra_auc, contra_top1_acc, contra_top5_acc  = self.pim(batch_repre) 
            distance_loss = self.bim(batch, batch_repre)
            loss_dict =  {'pim_loss':contrastive_loss, 
                        'bim_loss': distance_loss}   
            metric_dict = {'contra_auc': contra_auc, 
                        'contra_top1_acc': contra_top1_acc,
                        'contra_top5_acc': contra_top5_acc}
            
        elif self.cfg.train_mode == 'bim':
            distance_loss = self.bim(batch, batch_repre)
            loss_dict = {'bim_loss': distance_loss}
            metric_dict = {'distance_mse': distance_loss.view(-1)}

        elif self.cfg.train_mode == 'pim':
            contrastive_loss, contra_auc, contra_top1_acc, contra_top5_acc  = self.pim(batch_repre)
            loss_dict =  {'pim_loss':contrastive_loss}
            metric_dict = {'contra_auc': contra_auc, 
                        'contra_top1_acc': contra_top1_acc,
                        'contra_top5_acc': contra_top5_acc}

        else:
            raise NotImplementedError('Training mode not supported: %s' % self.cfg.train_mode)
            
        return loss_dict, metric_dict
        


