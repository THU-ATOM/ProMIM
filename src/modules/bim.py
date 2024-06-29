import torch.nn as nn
import torch
import torch.nn.functional as F
from src.modules.encoders.transformer_encoder_with_pair import TransformerEncoderWithPair
from unicore import utils
from src.utils.protein.constants import BBHeavyAtom

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class BIM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.cross_distance_project = NonLinearHead(
        cfg.encoder_embed_dim * 2 + cfg.encoder_attention_heads, 1, "relu"
        )

        self.pair_rep_compress = nn.Linear(cfg.encoder_embed_dim * 2, 1)

        self.concat_decoder = TransformerEncoderWithPair(
            encoder_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,     
            ffn_embed_dim=cfg.encoder_ffn_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )


    def crnet_forward(self, node_rep_1, pair_rep_1, node_rep_2, pair_rep_2, padding_mask):

        bsz = node_rep_1.size(0)
        head_num = self.cfg.encoder_attention_heads
        pro_sz_1 = node_rep_1.size(1)
        pro_sz_2 = node_rep_2.size(1)

        concat_rep = torch.cat(
            [node_rep_1, node_rep_2], dim=-2
        ) 
        concat_mask = padding_mask

        concat_attn_bias = torch.zeros(
            bsz, pro_sz_1 + pro_sz_2, pro_sz_1 + pro_sz_2
        ).type_as(
            concat_rep
        )
        concat_attn_bias[:, :pro_sz_1, :pro_sz_1] = pair_rep_1
        concat_attn_bias[:, -pro_sz_2:, -pro_sz_2:] = pair_rep_2
        concat_attn_bias = concat_attn_bias.repeat(head_num, 1, 1)

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias

        for i in range(self.cfg.recycling): 
            decoder_outputs = self.concat_decoder(
                decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep
            )
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i != (self.cfg.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                    -1, pro_sz_1 + pro_sz_2, pro_sz_1 + pro_sz_2
                )

        pro_decoder_1 = decoder_rep[:, :pro_sz_1]
        pro_decoder_2 = decoder_rep[:, pro_sz_1:]

        return pro_decoder_1, pro_decoder_2, decoder_pair_rep


    def pair_rep_construct(self, node_rep):

        batch_size, num_nodes, rep_size = node_rep.size()
        reshaped_node_rep = node_rep.clone().view(batch_size, num_nodes, 1, rep_size)
        repeated_node_rep = reshaped_node_rep.expand(batch_size, num_nodes, num_nodes, rep_size)
        pair_rep = torch.cat([repeated_node_rep, repeated_node_rep.transpose(1, 2)], dim=-1)
        pair_rep = self.pair_rep_compress(pair_rep).squeeze(-1)

        return pair_rep


    def forward(self, batch, batch_repre):
        # attain patch size
        patch_size = int(batch_repre.shape[1] / 2)

        # node-level   [batch, patch_size, emb_size]
        node_rep_1 = batch_repre[:,:patch_size,:]
        node_rep_2 = batch_repre[:,patch_size:,:] 

        pair_rep_1 = self.pair_rep_construct(node_rep_1)
        pair_rep_2 = self.pair_rep_construct(node_rep_2)

        # padding mask [batch, patch_size*2]
        padding_mask = batch['mask']

        # BindNet Transformer
        pro_decoder_1, pro_decoder_2, decoder_pair_rep = self.crnet_forward(node_rep_1, pair_rep_1, node_rep_2, pair_rep_2, padding_mask)

        protein_pair_decoder_rep = (
            decoder_pair_rep[:, :patch_size, patch_size:, :]
            + decoder_pair_rep[:, patch_size:, :patch_size, :].transpose(1, 2)
        ) / 2.0
        protein_pair_decoder_rep[protein_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                protein_pair_decoder_rep,
                pro_decoder_1.unsqueeze(-2).repeat(1, 1, patch_size, 1),
                pro_decoder_2.unsqueeze(-3).repeat(1, patch_size, 1, 1),
            ],
            dim=-1,
        )

        cross_distance_predict = (
        F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # [batch, patch_size, patch_size]

        # distance mask
        pro_padding_mask_1 = padding_mask[:,:patch_size]
        pro_padding_mask_2 = padding_mask[:,patch_size:]
        distance_mask = torch.logical_or(pro_padding_mask_1.unsqueeze(2), pro_padding_mask_2.unsqueeze(1)).ne(True)

        # construct label for regression
        pos_CA = batch['pos_atoms'][:,:,BBHeavyAtom.CA]
        pos_CA_1 = pos_CA[:,:patch_size,:]
        pos_CA_2 = pos_CA[:,patch_size:,:]
        cross_distance = torch.norm(pos_CA_1.unsqueeze(2) - pos_CA_2.unsqueeze(1), dim=-1)
        
        # regression loss
        distance_predict = cross_distance_predict[distance_mask]
        distance_target = cross_distance[distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), distance_target.float(), reduction="mean"
        )

        return distance_loss

