from uav.config import get_config
import math
from functools import reduce, lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F

@lru_cache(maxsize=32)
def _spatial_roll_index(height, width, shifts, device):
    # A bounded, shared cache across transformer layers; no media tensors retained.
    y = (torch.arange(height, device=device) - shifts[0]) % height
    x = (torch.arange(width, device=device) - shifts[1]) % width
    return (y[:, None] * width + x[None, :]).reshape(-1)


def spatial_roll(x, shifts):
    """Exact two-axis roll for BTHWC, with one gather instead of two copies."""
    height, width = x.shape[2:4]
    index = _spatial_roll_index(height, width, tuple(shifts), x.device)
    return x.flatten(2, 3).index_select(2, index).view_as(x)


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


@lru_cache(maxsize=32)
def _fold_normalizer(output_size, kernel_size, padding, stride, device, dtype):
    n = math.prod((output_size[i] + 2*padding[i] - kernel_size[i]) // stride[i] + 1
                  for i in range(2))
    return F.fold(torch.ones(1, math.prod(kernel_size), n, device=device, dtype=dtype),
                  output_size=output_size, kernel_size=kernel_size, padding=padding, stride=stride)


def _ffn_frame_forward(x, normalizer, weight1, bias1, weight2, bias2,
                       output_size, kernel, padding, stride):
    # Frames are independent in this block. Keep frame count separate so only
    # that dimension is dynamic; spatial fold/unfold loops remain specialized.
    x=F.linear(x,weight1,bias1)
    x=F.fold(x.permute(0,2,1),output_size=output_size,
             kernel_size=kernel,padding=padding,stride=stride)
    x=F.unfold(x/normalizer,kernel_size=kernel,padding=padding,stride=stride)
    x=x.permute(0,2,1).contiguous()
    return F.linear(F.gelu(x),weight2,bias2)


_ffn_signatures = set()
_ffn_calls = {'compiled': 0, 'fallback': 0}


def ffn_compile_profile():
    return dict(_ffn_calls, signatures=len(_ffn_signatures), capacity=8)


@lru_cache(maxsize=1)
def _compiled_ffn():
    # Share compiled code across layers; weights remain ordinary tensor inputs.
    return torch.compile(_ffn_frame_forward, fullgraph=True, dynamic=False,
                         options={'emulate_precision_casts': True})


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size']) # 49

    def forward(self, x, output_size):
        if (not self.training and x.is_cuda and
                get_config().propainter_compile_ffn):
            kernel=tuple(self.t2t_params['kernel_size'])
            padding=tuple(self.t2t_params['padding'])
            stride=tuple(self.t2t_params['stride'])
            normalizer=_fold_normalizer(tuple(output_size),kernel,padding,stride,x.device,x.dtype)
            n_vecs=((output_size[0]+2*padding[0]-kernel[0])//stride[0]+1) * (
                    (output_size[1]+2*padding[1]-kernel[1])//stride[1]+1)
            frames=x.reshape(-1,n_vecs,x.shape[-1])
            key=(tuple(frames.shape[1:]),tuple(frames.stride()),x.dtype,x.device,
                 tuple(output_size),kernel,padding,stride,
                 tuple(self.fc1[0].weight.shape),tuple(self.fc2[1].weight.shape),
                 torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'))
            if key not in _ffn_signatures and len(_ffn_signatures)<8:
                _ffn_signatures.add(key)
            use_compiled=key in _ffn_signatures
            _ffn_calls['compiled' if use_compiled else 'fallback']+=1
            operation=_compiled_ffn() if use_compiled else _ffn_frame_forward
            if use_compiled:
                torch._dynamo.mark_dynamic(frames,0)
            result=operation(frames,normalizer,self.fc1[0].weight,self.fc1[0].bias,
                self.fc2[1].weight,self.fc2[1].bias,tuple(output_size),kernel,padding,stride)
            return result.reshape_as(x)
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        if not self.training:
            normalizer = _fold_normalizer(tuple(output_size), tuple(self.t2t_params['kernel_size']),
                tuple(self.t2t_params['padding']), tuple(self.t2t_params['stride']), x.device, x.dtype)
        else:
            normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
            normalizer = F.fold(normalizer,
                                output_size=output_size,
                                kernel_size=self.t2t_params['kernel_size'],
                                padding=self.t2t_params['padding'],
                                stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.fc2(x)
        return x


def window_partition(x, window_size, n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C//n_head)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows

@lru_cache(maxsize=32)
def _window_pixel_indices(h,w,wh,ww,device):
    windows=torch.arange((h//wh)*(w//ww),device=device)
    local=torch.arange(wh*ww,device=device)
    return (windows[:,None]//(w//ww)*wh+local[None]//ww)*w+windows[:,None]%(w//ww)*ww+local[None]%ww


def selected_window_tokens(x,window_size,heads,selected):
    """Gather selected local windows without materializing every window."""
    b,t,h,w,c=x.shape;wh,ww=window_size
    index=_window_pixel_indices(h,w,wh,ww,x.device).index_select(0,selected)
    return x.flatten(2,3).index_select(2,index.flatten()).view(b,t,len(selected),wh*ww,heads,c//heads).permute(0,2,4,1,3,5).contiguous()


def context_tokens(module, x, selected=None, times=None):
    b,t,h,w,c=x.shape
    wh,ww=module.window_size;eh,ew=module.expand_size
    nh,nw=h//wh,w//ww;heads=module.n_head;d=c//heads
    if not any(module.expand_size):
        if times is not None:x=x[:,times];t=len(times)
        windows=window_partition(x, module.window_size, heads).view(b,nh*nw,heads,t,wh*ww,d)
        return windows if selected is None else windows.index_select(1,selected)
    valid = module.valid_ind_rolled
    version = None if torch.is_inference(valid) else valid._version
    key=(h,w,x.device,tuple(module.window_size),tuple(module.expand_size),id(valid),version)
    cache=getattr(module,'_context_index_cache',None)
    if cache is None:cache={};module._context_index_cache=cache
    if key not in cache:
        # Reuse exact checkpoint buffer ordering, including repeated source pixels.
        valid=module.valid_ind_rolled
        direction=valid//(wh*ww);position=valid%(wh*ww)
        shifts=valid.new_tensor([[-eh,-ew],[-eh,ew],[eh,-ew],[eh,ew]])
        local=torch.arange(wh*ww,device=x.device)
        ys=torch.cat((local//ww,position//ww-shifts[direction,0]))
        xs=torch.cat((local%ww,position%ww-shifts[direction,1]))
        windows=torch.arange(nh*nw,device=x.device)
        yy=(windows[:,None]//nw*wh+ys[None,:])%h
        xx=(windows[:,None]%nw*ww+xs[None,:])%w
        index=(yy*w+xx)
        if len(cache)>=32:cache.pop(next(iter(cache)))
        cache[key]=index
    index=cache[key]
    if selected is not None:
        index=index.index_select(0,selected)
    count,n=index.shape
    if times is not None:
        # Gather only the requested temporal/spatial context. Selecting frames
        # first copies every pixel, most of which is discarded by this gather.
        index=(times[:,None,None]*(h*w)+index[None]).reshape(-1)
        return x.flatten(1,3).index_select(1,index).view(b,len(times),count,n,heads,d).permute(0,2,4,1,3,5).contiguous()
    return x.flatten(2,3).index_select(2,index.reshape(-1)).view(b,t,count,n,heads,d).permute(0,2,4,1,3,5).contiguous()


class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0.,
                pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))


    def _load_from_state_dict(self,*args,**kwargs):
        self._packed_projection=None
        return super()._load_from_state_dict(*args,**kwargs)

    def _projection_weights(self):
        layers=(self.query,self.key,self.value)
        parameters=tuple(p for layer in layers for p in (layer.weight,layer.bias) if p is not None)
        # Do not retain derived weights during training; invalidate on checkpoint
        # reload, device/dtype conversion or any ordinary in-place weight update.
        signature=tuple((id(p),None if torch.is_inference(p) else p._version,p.device,p.dtype) for p in parameters)
        cached=getattr(self,'_packed_projection',None)
        if cached is None or cached[0]!=signature:
            weight=torch.cat([layer.weight for layer in layers])
            bias=None if layers[0].bias is None else torch.cat([layer.bias for layer in layers])
            cached=(signature,weight,bias)
            self._packed_projection=cached
        return cached[1:]

    def project(self,x,pooled=False):
        if self.training or torch.is_grad_enabled() or not get_config().propainter_pack_qkv:
            return (self.key(x),self.value(x)) if pooled else (self.query(x),self.key(x),self.value(x))
        weight,bias=self._projection_weights()
        if pooled:
            dim=self.query.out_features
            return F.linear(x,weight[dim:],None if bias is None else bias[dim:]).chunk(2,dim=-1)
        return F.linear(x,weight,bias).chunk(3,dim=-1)

    def prepare_indices(self, mask, spatial_size):
        """Classify windows once; all attention layers share the same input mask."""
        h, w = spatial_size
        wh, ww = self.window_size
        ph, pw = (-h) % wh, (-w) % ww
        if ph or pw:
            mask = F.pad(mask, (0, 0, 0, pw, 0, ph, 0, 0))
        b, lt = mask.shape[:2]
        pooled = self.max_pool(mask.reshape(b * lt, h + ph, w + pw))
        pooled = pooled.view(b, lt, -1).sum(1)
        return [(pooled[i].nonzero(as_tuple=False).view(-1),
                 (pooled[i] == 0).nonzero(as_tuple=False).view(-1)) for i in range(b)]

    def forward(self, x, mask=None, T_ind=None, attn_mask=None, prepared_indices=None):
        b, t, h, w, c = x.shape # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0] # 20
        new_w = n_ww * self.window_size[1] # 36
        pad_r = new_w - w
        pad_b = new_h - h
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q,k,v = self.project(x)
        win_q = window_partition(q, self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        sparse_context = not self.training and get_config().propainter_sparse_context
        if not sparse_context:
            win_k = context_tokens(self, k)
            win_v = context_tokens(self, v)

        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b*t, new_h, new_w, c).permute(0,3,1,2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0,2,3,1).view(b, t, p_h, p_w, c)
            projected_k,projected_v=self.project(pool_x,pooled=True)
            # pool_k
            # Share pooled tokens across windows; cat materializes them once.
            pool_k = projected_k.reshape(b, t, p_h*p_w, self.n_head, c_head).permute(0, 3, 1, 2, 4)
            pool_k = pool_k[:, None].expand(-1, n_wh*n_ww, -1, -1, -1, -1)
            if not sparse_context:
                win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            # Share pooled tokens across windows; cat materializes them once.
            pool_v = projected_v.reshape(b, t, p_h*p_w, self.n_head, c_head).permute(0, 3, 1, 2, 4)
            pool_v = pool_v[:, None].expand(-1, n_wh*n_ww, -1, -1, -1, -1)
            if not sparse_context:
                win_v = torch.cat((win_v, pool_v), dim=4)

        # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
        # Masked and unmasked indices partition every window; both branches
        # overwrite their full output, so no separate zero fill is needed.
        out = torch.empty_like(win_q)
        l_t = mask.size(1)

        if prepared_indices is None:
            prepared_indices = self.prepare_indices(mask, (h, w))
        for i in range(win_q.shape[0]):
            ### For masked windows
            mask_ind_i = prepared_indices[i][0]
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.n_head, t*w_h*w_w, c_head)
                if sparse_context:
                    win_k_t = context_tokens(self,k[i:i+1],mask_ind_i,T_ind)[0]
                    win_v_t = context_tokens(self,v[i:i+1],mask_ind_i,T_ind)[0]
                    if self.pooling_token:
                        # Every spatial window shares the same pooled tokens.
                        # Select time once, then broadcast instead of copying
                        # every window before temporal selection.
                        pk, pv = pool_k[i,0], pool_v[i,0]
                        if T_ind is not None:
                            pk, pv = pk[:,T_ind], pv[:,T_ind]
                        pk=pk.unsqueeze(0).expand(mask_n,-1,-1,-1,-1)
                        pv=pv.unsqueeze(0).expand(mask_n,-1,-1,-1,-1)
                        win_k_t = torch.cat((win_k_t,pk),dim=3)
                        win_v_t = torch.cat((win_v_t,pv),dim=3)
                else:
                    win_k_t = win_k[i, mask_ind_i]
                    win_v_t = win_v[i, mask_ind_i]
                # mask out key and value
                if T_ind is not None:
                    # key [n_wh*n_ww, n_head, t, w_h*w_w, c_head]
                    if not sparse_context:
                        win_k_t = win_k_t[:, :, T_ind.view(-1)]
                        win_v_t = win_v_t[:, :, T_ind.view(-1)]
                    win_k_t = win_k_t.reshape(mask_n, self.n_head, -1, c_head)
                    win_v_t = win_v_t.reshape(mask_n, self.n_head, -1, c_head)
                else:
                    win_k_t = win_k_t.reshape(mask_n, self.n_head, -1, c_head)
                    win_v_t = win_v_t.reshape(mask_n, self.n_head, -1, c_head)

                if not self.training and get_config().propainter_sdpa:
                    y_t = F.scaled_dot_product_attention(win_q_t, win_k_t, win_v_t)
                else:
                    att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                    att_t = F.softmax(att_t, dim=-1)
                    att_t = self.attn_drop(att_t)
                    y_t = att_t @ win_v_t

                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h*w_w, c_head)

            ### For unmasked windows
            unmask_ind_i = prepared_indices[i][1]
            if not len(unmask_ind_i):continue
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            win_q_s = win_q[i, unmask_ind_i]
            if sparse_context:
                win_k_s = selected_window_tokens(k[i:i+1],self.window_size,self.n_head,unmask_ind_i)[0]
                win_v_s = selected_window_tokens(v[i:i+1],self.window_size,self.n_head,unmask_ind_i)[0]
            else:
                win_k_s = win_k[i, unmask_ind_i, :, :, :w_h*w_w]
                win_v_s = win_v[i, unmask_ind_i, :, :, :w_h*w_w]

            if not self.training and get_config().propainter_sdpa:
                shape = win_q_s.shape
                qs = win_q_s.flatten(0, 2)
                ks = win_k_s.flatten(0, 2)
                vs = win_v_s.flatten(0, 2)
                y_s = F.scaled_dot_product_attention(qs.unsqueeze(0), ks.unsqueeze(0), vs.unsqueeze(0)).reshape(shape)
            else:
                att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
                att_s = F.softmax(att_s, dim=-1)
                att_s = self.attn_drop(att_s)
                y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)


        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        # output projection
        out = self.proj_drop(self.proj(out))
        return out


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size,
                norm_layer=nn.LayerNorm, t2t_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self, x, fold_x_size, mask=None, T_ind=None, prepared_indices=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape # 20 36

        shortcut = x
        x = self.norm1(x)
        att_x = self.attention(x, mask, T_ind, prepared_indices=prepared_indices)

        # FFN
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None):
        super().__init__()
        blocks = []
        for i in range(depths):
             blocks.append(
                TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t_params=t2t_params)
             )
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation, device=x.device) for i in range(t_dilation)] * (self.depths // t_dilation)

        # Share only within this invocation, never across changing masks.
        attention = self.transformer[0].attention
        shared_layout = all(block.attention.window_size == attention.window_size
                            for block in self.transformer)
        prepared = attention.prepare_indices(l_mask, x.shape[2:4]) if shared_layout else None
        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, l_mask, T_ind[i], prepared_indices=prepared)

        return x
