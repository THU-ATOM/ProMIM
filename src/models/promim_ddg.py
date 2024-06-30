import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.encoders.single import PerResidueEncoder
from src.modules.encoders.pair import ResiduePairEncoder
from src.modules.encoders.attn import GAEncoder
from src.utils.protein.constants import BBHeavyAtom
from .promim import ProMIM
from .rde import CircularSplineRotamerDensityEstimator


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.relu(x + residual)
        return x


class DDG_Network(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # PIM, BIM
        ckpt = torch.load(cfg.promim_ckpt, map_location='cpu')
        self.promim = ProMIM(ckpt['config'].model)
        self.promim.load_state_dict(ckpt['model'])
        for p in self.promim.parameters():
            p.requires_grad_(False)

        # SIM
        rde_ckpt = torch.load(cfg.rde_ckpt, map_location='cpu')
        self.rde = CircularSplineRotamerDensityEstimator(rde_ckpt['config'].model)
        self.rde.load_state_dict(rde_ckpt['model'])
        for p in self.rde.parameters():
            p.requires_grad_(False)        

        dim = ckpt['config'].model.encoder.node_feat_dim

        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.encoder.node_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )
        self.single_fusion = nn.Sequential(
            nn.Linear(3*dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), 
        )
        self.mut_bias = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = dim,
            padding_idx = 0,
        )

        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(**cfg.encoder)

        # Pred
        self.ddg_readout = nn.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim),
            nn.Linear(dim, 1)
        )


    def _encode_promim(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.promim.encode(batch)
        
    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)
        

    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x_single = self.single_encoder(
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,
        )

        x_pret = self._encode_promim(batch)
        x_pret_rde= self._encode_rde(batch)

        x = self.single_fusion(torch.cat([x_single, x_pret, x_pret_rde], dim=-1))

        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b

        z = self.pair_encoder(
            aa = batch['aa'], 
            res_nb = batch['res_nb'], chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_atoms'], mask_atoms = batch['mask_atoms'],
        )

        x = self.attn_encoder(
            pos_atoms = batch['pos_atoms'], 
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
        batch['transformed_pos_atoms'] = batch['pos_atoms']

        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
            
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2
        
        loss_dict = {
            'regression': loss,
        }
        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'],
        }
        
        return loss_dict, out_dict