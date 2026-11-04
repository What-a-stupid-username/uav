''' Towards An End-to-End Framework for Video Inpainting
'''
from uav.config import get_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from einops import rearrange

from uav.propainter.video.model.modules.base_module import BaseNetwork
from uav.propainter.video.model.modules.sparse_transformer import TemporalSparseTransformerBlock, SoftSplit, SoftComp
from uav.propainter.video.model.modules.spectral_norm import spectral_norm as _spectral_norm
from uav.propainter.video.model.modules.flow_loss_utils import flow_warp,normalized_flow_grid
from uav.propainter.video.model.modules.deformconv import ModulatedDeformConv2d

from .misc import constant_init


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5, *, grid=None):
    flow_bw_warped = (flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1)) if grid is None
                      else F.grid_sample(flow_bw,grid,align_corners=True))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw


def _alignment_offset(out, flow, max_residue_magnitude):
    # The first two chunks are adjacent views: use their common prefix instead
    # of allocating/copying them through cat. Broadcast the two flow channels
    # across offset groups rather than materializing repeat's full tensor.
    channels = out.shape[1] // 3
    offset = max_residue_magnitude * torch.tanh(out[:, :2 * channels])
    b, _, h, w = offset.shape
    offset = (offset.reshape(b, channels, 2, h, w) +
              flow.flip(1).unsqueeze(1)).reshape(b, 2 * channels, h, w)
    return offset, torch.sigmoid(out[:, 2 * channels:])


class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        offset, mask = _alignment_offset(out, flow, self.max_residue_magnitude)

        if (not self.training and get_config().propainter_stream_deform):
            from uav.deform_conv import deform_conv2d
            return deform_conv2d(x,offset,self.weight,self.bias,padding=self.padding,mask=mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias,
                                             self.stride, self.padding,
                                             self.dilation, mask)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

    def binary_mask(self, mask, th=0.1):
        mask.masked_fill_(mask > th, 1)
        mask.masked_fill_(mask <= th, 0)
        # return mask.float()
        return mask.to(mask)

    def forward(self, x, flows_forward, flows_backward, mask, interpolation='bilinear', *, fused_only=False):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        batched_valid = None
        sampling_grids=None
        if (not self.training and t > 1):
            forward = flows_forward.reshape(-1, 2, h, w)
            backward = flows_backward.reshape(-1, 2, h, w)
            if not torch.is_grad_enabled() and forward.dtype==backward.dtype==x.dtype:
                sampling_grids={
                    'backward_1': normalized_flow_grid(forward.permute(0,2,3,1),x.dtype),
                    'forward_1': normalized_flow_grid(backward.permute(0,2,3,1),x.dtype)}
            batched_valid = {
                'backward_1': fbConsistencyCheck(forward, backward,grid=None if sampling_grids is None else sampling_grids['backward_1']).view(b, t-1, 1, h, w),
                'forward_1': fbConsistencyCheck(backward, forward,grid=None if sampling_grids is None else sampling_grids['forward_1']).view(b, t-1, 1, h, w),
            }
            if sampling_grids is not None:
                sampling_grids={name:grid.view(b,t-1,h,w,2) for name,grid in sampling_grids.items()}
        feats, masks = {}, {}
        feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]
        masks['input'] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ['backward_1', 'forward_1']
        cache_list = ['input'] + prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = (fbConsistencyCheck(flow_prop, flow_check) if batched_valid is None
                                       else batched_valid[module_name][:, flow_idx[i]])
                    grid=None if sampling_grids is None else sampling_grids[module_name][:,flow_idx[i]]
                    feat_warped = (flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation) if grid is None
                                   else F.grid_sample(feat_prop,grid,mode=interpolation,align_corners=True))

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = (flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1)) if grid is None
                                           else F.grid_sample(mask_prop,grid,align_corners=True))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)
                        # Optional diagnostic callback; absent in normal inference.
                        observer = getattr(self, 'debug_observer', None)
                        if observer is not None:
                            observer(module_name, idx, mask_current, flow_vaild_mask, mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current * flow_vaild_mask * (1 - mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1 - union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current * (1 - (flow_vaild_mask * (1 - mask_prop_valid))))

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        if self.learnable and fused_only and not self.training:
            # Stack directly into the convolution input, avoiding intermediate
            # backward/forward tensors followed by another concatenation.
            fused_input=x.new_empty(b,t,2*c+2,h,w)
            torch.stack(feats['backward_1'],dim=1,out=fused_input[:,:,:c])
            torch.stack(feats['forward_1'],dim=1,out=fused_input[:,:,c:2*c])
            fused_input[:,:,2*c:].copy_(mask)
            fuse_input=fused_input.view(b*t,2*c+2,h,w)
            outputs=self.fuse(fuse_input)+x.view(-1,c,h,w)
            return None,None,outputs.view(b,t,c,h,w),None

        outputs_b = torch.stack(feats['backward_1'], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats['forward_1'], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.view(-1, 2, h, w)
            masks_b, masks_f = None, None
            fuse_input=torch.cat([outputs_b, outputs_f, mask_in],dim=1)
            outputs = self.fuse(fuse_input) + x.view(-1, c, h, w)
        else:
            masks_b = torch.stack(masks['backward_1'], dim=1)
            masks_f = torch.stack(masks['forward_1'], dim=1)
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
            outputs.view(b, -1, c, h, w), masks_f


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, model_path=None):
        super(InpaintGenerator, self).__init__()
        channel = 128
        hidden = 512

        # encoder
        self.encoder = Encoder()

        # decoder
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)

        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(dim=hidden,
                                                           n_head=num_heads,
                                                           window_size=window_size,
                                                           pool_size=pool_size,
                                                           depths=depths,
                                                           t2t_params=t2t_params)
        if init_weights:
            self.init_weights()

        if model_path is not None:
            print('Pretrained ProPainter has loaded...')
            ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)

        # print network parameter number
        self.print_network()

    def img_propagation(self, masked_frames, completed_flows, masks, interpolation='nearest'):
        _, _, prop_frames, updated_masks = self.img_prop_module(masked_frames, completed_flows[0], completed_flows[1],
                                                                masks, interpolation)
        return prop_frames, updated_masks

    def encode_frames(self, frames, masks_in, masks_updated):
        """Frame-independent encoder, reusable within one propagated clip."""
        b,t,_,h,w=frames.shape
        features=self.encoder(torch.cat((frames.reshape(b*t,3,h,w),
            masks_in.reshape(b*t,1,h,w),masks_updated.reshape(b*t,1,h,w)),dim=1))
        return features.reshape(b,t,*features.shape[1:])

    def forward(self, masked_frames, completed_flows, masks_in, masks_updated, num_local_frames,
                interpolation='bilinear', t_dilation=2, encoded_features=None, output_indices=None,skip_feature_propagation=False,return_decoder_features=False):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """

        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.size()
        # RGB values are consumed only by encode_frames. With cached features,
        # external reference RGB need not be concatenated for every window.
        if encoded_features is not None:
            t = encoded_features.size(1)

        # extracting features
        if encoded_features is None:
            encoded=self.encode_frames(masked_frames,masks_in,masks_updated)
            enc_feat=encoded.reshape(b*t,*encoded.shape[2:])
        else:
            if self.training:raise ValueError('encoded feature reuse is inference-only')
            enc_feat=encoded_features.reshape(b*t,*encoded_features.shape[2:])
        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        prepared=None if self.training or torch.is_grad_enabled() else getattr(self.feat_prop_module.forward,'prepared',None)
        if prepared is None and not skip_feature_propagation:
            ds_flows_f = F.interpolate(completed_flows[0].view(-1, 2, ori_h, ori_w), scale_factor=1 / 4, mode='bilinear',
                                       align_corners=False).view(b, l_t - 1, 2, h, w) / 4.0
            ds_flows_b = F.interpolate(completed_flows[1].view(-1, 2, ori_h, ori_w), scale_factor=1 / 4, mode='bilinear',
                                       align_corners=False).view(b, l_t - 1, 2, h, w) / 4.0
        mask_frames = t if self.training else l_t
        ds_mask_in = F.interpolate(masks_in[:, :mask_frames].reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4, mode='nearest').view(b, mask_frames,
                                                                                                                   1, h,
                                                                                                                   w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        if prepared is None and not skip_feature_propagation:
            ds_mask_updated_local = F.interpolate(masks_updated[:, :l_t].reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4,
                                                  mode='nearest').view(b, l_t, 1, h, w)

        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, l_t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))

        if skip_feature_propagation:
            pass
        elif prepared is None:
            prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
            _, _, local_feat, _ = self.feat_prop_module(local_feat, ds_flows_f, ds_flows_b, prop_mask_in, interpolation)
        else:
            _, _, local_feat, _ = prepared(local_feat,completed_flows,masks_in,masks_updated,interpolation)
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        mask_pool_l = rearrange(mask_pool_l, 'b t c h w -> b t h w c').contiguous()
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        if not self.training and output_indices is not None:
            # Attention still uses every frame; SoftComp is frame-independent.
            # Reconstruct only frames requested by the caller, as for the decoder.
            selection=output_indices
            if output_indices and all(j==output_indices[0]+i for i,j in enumerate(output_indices)):
                selection=slice(output_indices[0],output_indices[-1]+1)
            trans_feat = self.sc(trans_feat[:, selection], len(output_indices), fold_feat_size)
            trans_feat = trans_feat.view(b, len(output_indices), -1, h, w)
            decode_feat = enc_feat[:, selection] + trans_feat
            if return_decoder_features:return decode_feat
            output = self.decoder(decode_feat.reshape(-1, c, h, w))
            return torch.tanh(output).view(b, len(output_indices), 3, ori_h, ori_w)

        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, c, h, w))
            output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        else:
            decode_feat=enc_feat[:, :l_t] if output_indices is None else enc_feat[:,output_indices]
            if return_decoder_features:return decode_feat
            output = self.decoder(decode_feat.reshape(-1, c, h, w))
            output = torch.tanh(output).view(b, decode_feat.shape[1], 3, ori_h, ori_w)

        return output


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################
class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


class Discriminator_2D(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator_2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(1, 5, 5),
                      stride=(1, 2, 2),
                      padding=(0, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
