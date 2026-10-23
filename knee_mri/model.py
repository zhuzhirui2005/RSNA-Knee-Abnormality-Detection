# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""2.5D ConvNeXt with class-specific spatial and study attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .constants import LABELS
C = len(LABELS)

class KneeClassifier(nn.Module):

    def __init__(self, backbone, init, hidden=384, drop=0.1, freeze_blocks=16, crop=256, state_spec=None, in_chans=3, spatial=False, extent=False):
        super().__init__()
        import timm
        self.backbone = backbone
        self.crop = int(crop)
        self.in_chans = int(in_chans)
        self.spatial = bool(spatial)
        self.extent = bool(extent)
        self._ext_area = None
        self.enc = self._timm_encoder(backbone, init, self.in_chans)
        self.has_bn = any((isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.enc.modules()))
        self._build_head(hidden, drop)
        self.state_spec = state_spec
        self.has_state = False
        if self.spatial:
            d = self.enc.num_features
            self.sp_key = nn.Linear(d, 128)
            self.sp_pos = nn.Linear(16, 128, bias=False)
            nn.init.zeros_(self.sp_pos.weight)
            self.sp_q = nn.Parameter(torch.zeros(C, 128))
        if self.extent:
            self.ext_u = nn.Parameter(torch.zeros(C, hidden))
            self.ext_b = nn.Parameter(torch.full((C,), -4.0))
            self.ext_gamma = nn.Parameter(torch.zeros(C))
            self.ext_view = nn.Parameter(torch.zeros(C, 6))
            em = torch.zeros(C)
            for l in ('Medial OA', 'Lateral OA', 'PF OA'):
                em[LABELS.index(l)] = 1.0
            self.register_buffer('ext_mask', em)

    def _timm_encoder(self, backbone, init, in_chans=3):
        import timm
        is_vit = backbone.startswith('vit_')
        kw = dict(pretrained=init == 'timm', num_classes=0, in_chans=in_chans, drop_path_rate=0.1 if 'convnext' in backbone or is_vit else 0.0)
        if is_vit:
            kw['dynamic_img_size'] = True
        else:
            kw['global_pool'] = 'avg'
        if backbone.startswith(('coatnet', 'maxvit', 'maxxvit')):
            kw['img_size'] = self.crop
        if self.spatial:
            kw['global_pool'] = ''
        enc = timm.create_model(backbone, **kw)
        if init and init != 'timm':
            if str(init).endswith('.safetensors'):
                from safetensors.torch import load as load_bytes
                with open(str(init), 'rb') as fh:
                    sd = load_bytes(fh.read())
            else:
                sd = torch.load(init, map_location='cpu', weights_only=True)
                sd = sd.get('state_dict', sd.get('model', sd)) if isinstance(sd, dict) else sd
            if in_chans != 3:
                for k, w in list(sd.items()):
                    if w.ndim == 4 and w.shape[1] == 3 and (k in enc.state_dict()) and (enc.state_dict()[k].shape[1] == in_chans):
                        sd[k] = w.mean(1, keepdim=True).repeat(1, in_chans, 1, 1) * (3.0 / in_chans)
                        print('inflated %s %s -> %s' % (k, tuple(w.shape), tuple(sd[k].shape)), flush=True)
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            print('init %s: missing %d unexpected %d' % (init, len(missing), len(unexpected)), flush=True)
        self.sp_head_norm = None
        if self.spatial and isinstance(getattr(getattr(enc, 'head', None), 'norm', None), nn.Module) and (not isinstance(enc.head.norm, nn.Identity)):
            self.sp_head_norm = enc.head.norm
            enc.head.norm = nn.Identity()
        return enc

    def _build_head(self, hidden, drop):
        d = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden))
        self.slot = nn.Embedding(10, hidden)
        self.view = nn.Embedding(6, hidden)
        self.geom = nn.Linear(4, hidden, bias=False)
        for m in (self.slot, self.view):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.geom.weight)
        self.drop = nn.Dropout(drop)
        self.win_proj = nn.Linear(hidden, hidden)
        self.q_win = nn.Parameter(torch.randn(C, hidden) * 0.02)
        self.out_norm = nn.LayerNorm(hidden)
        self.cls_w = nn.Parameter(torch.randn(C, hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(C))
        if self.in_chans == 3:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.full((1, self.in_chans, 1, 1), 0.449))
            self.register_buffer('std', torch.full((1, self.in_chans, 1, 1), 0.226))

    def spatial_pool(self, x):
        """Encoder output with its spatial grid ([N, d, h, w] CNN map or [N, T, d] ViT tokens) -> per-class window features
        [N, C, d]: class c weighs the window's positions with its own query (sp_q[c]; zero at init = the plain average)
        over keys tanh(W f(p) + W_pos pe(p)), so a class can read its own structure inside a window shared with others."""
        if x.ndim == 4:
            h, w = x.shape[-2:]
            f = x.flatten(2).transpose(1, 2)
        else:
            f = x[:, getattr(self.enc, 'num_prefix_tokens', 0):]
            h = w = int(round(f.shape[1] ** 0.5))
            assert h * w == f.shape[1], 'non-square ViT token grid'
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=f.device), torch.linspace(-1, 1, w, device=f.device), indexing='ij')
        fr = torch.tensor([1.0, 2.0, 4.0, 8.0], device=f.device) * (math.pi / 2)
        yy, xx = (yy.reshape(-1, 1) * fr, xx.reshape(-1, 1) * fr)
        pe = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1)
        kin = F.layer_norm(f.float(), (f.shape[-1],)).to(f.dtype)
        k = torch.tanh(self.sp_key(kin) + self.sp_pos(pe.to(f.dtype))[None])
        att = torch.einsum('npk,ck->npc', k, self.sp_q.to(k.dtype)).float().softmax(1).to(f.dtype)
        g = torch.einsum('npc,npd->ncd', att, f)
        if self.sp_head_norm is not None:
            n_ = self.sp_head_norm
            g = F.layer_norm(g.float(), (g.shape[-1],), n_.weight, n_.bias, n_.eps).to(f.dtype)
        return g

    def extent_area(self, e, meta, mask):
        """e [B, W, C] window defect logits -> A [B, C] (mm^2, an area proxy): inside each series the mean sigmoid(e) of
        each view's windows, combined over the views present by a learned per-(class, view) softmax (uniform at init: a
        compartment sits inside only some of the locals), times the series' slab thickness (slices x spacing) = mm of
        slab with defect evidence; per plane the max over series; then sagittal x coronal (tibiofemoral OA) or axial x
        sagittal (PF OA); other classes 0."""
        V = 6
        sid = meta[..., 6].long()
        S_MAX = max(16, int(sid.max()) + 1)
        vid = meta[..., 1].long().clamp(0, V - 1)
        q = torch.sigmoid(e.float()) * mask[..., None]
        ohv = F.one_hot(sid * V + vid, S_MAX * V).float() * mask[..., None]
        cntv = ohv.sum(1)
        mean_v = (torch.einsum('bwk,bwc->bkc', ohv, q) / cntv.clamp_min(1)[..., None]).view(len(e), S_MAX, V, -1)
        present = (cntv > 0).view(len(e), S_MAX, V, 1)
        wv = self.ext_view.float().t()[None, None].masked_fill(~present, -10000.0).softmax(2) * present
        mean = (wv * mean_v).sum(2)
        oh = F.one_hot(sid, S_MAX).float() * mask[..., None]
        cnt = oh.sum(1)
        span = torch.expm1(meta[..., 3].float()).clamp_min(0) * meta[..., 8].float()
        span_s = torch.einsum('bws,bw->bs', oh, span) / cnt.clamp_min(1)
        plane_s = (torch.einsum('bws,bw->bs', oh, meta[..., 7].float()) / cnt.clamp_min(1)).round()
        E = mean * span_s[..., None]
        per_plane = []
        for pl in (0, 1, 2):
            ok = ((plane_s == pl) & (cnt > 0))[..., None]
            per_plane.append(E.masked_fill(~ok, 0.0).amax(1))
        sag, cor, ax = per_plane
        area = torch.zeros_like(sag)
        for l in ('Medial OA', 'Lateral OA'):
            c = LABELS.index(l)
            area[:, c] = sag[:, c] * cor[:, c]
        c = LABELS.index('PF OA')
        area[:, c] = ax[:, c] * sag[:, c]
        return area

    def head(self, tok, meta, mask, return_aux=False):
        """Encoder output of the real windows [N, d] (or [N, C, d] with --spatial-attn) -> class logits [B, C] (and the
        state/binary logits when asked)."""
        B, W = mask.shape
        f = self.proj(tok.float())
        m = meta
        if self.spatial:
            emb = self.slot(m[..., 0].long().clamp(0, 9)) + self.view(m[..., 1].long().clamp(0, 5)) + self.geom(m[..., 2:6])
            t = f.new_zeros((B, W) + tuple(f.shape[1:]))
            t[mask] = f
            t = self.drop(t + emb[:, :, None, :])
            s2 = torch.einsum('bwch,ch->bwc', torch.tanh(self.win_proj(t)), self.q_win).float()
            s2 = s2.masked_fill(~mask[:, :, None], float('-inf'))
            a2 = s2.softmax(1).to(t.dtype)
            study = self.out_norm(torch.einsum('bwc,bwch->bch', a2, t))
            if self.extent:
                e_win = (t.float() * self.ext_u[None, None]).sum(-1) + self.ext_b
        else:
            t = f.new_zeros((B, W, f.shape[-1]))
            t[mask] = f
            t = t + self.slot(m[..., 0].long().clamp(0, 9)) + self.view(m[..., 1].long().clamp(0, 5)) + self.geom(m[..., 2:6])
            t = self.drop(t)
            s2 = torch.einsum('bwh,ch->bwc', torch.tanh(self.win_proj(t)), self.q_win).float()
            s2 = s2.masked_fill(~mask[:, :, None], float('-inf'))
            a2 = s2.softmax(1).to(t.dtype)
            study = self.out_norm(torch.einsum('bwc,bwh->bch', a2, t))
            if self.extent:
                e_win = torch.einsum('bwh,ch->bwc', t.float(), self.ext_u) + self.ext_b
        z = (study * self.cls_w).sum(-1).float() + self.bias
        if self.extent:
            area = self.extent_area(e_win, meta, mask)
            self._ext_area = area
            z = z + self.ext_gamma * self.ext_mask * torch.log1p(area / 100.0)
        return (z, None, None) if return_aux else z

    def forward(self, img, meta, mask, chunk=0, return_aux=False):
        x = img[mask]
        n = len(x)
        if n % 32 and (not self.training or not self.has_bn):
            x = torch.cat([x, x.new_zeros((32 - n % 32,) + tuple(x.shape[1:]))])
        feats = []
        step = chunk or len(x)
        for s in range(0, len(x), step):
            xs = x[s:s + step].float().div_(255.0)
            xs = ((xs - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            e = self.enc(xs)
            feats.append(self.spatial_pool(e) if self.spatial else e)
        return self.head(torch.cat(feats)[:n], meta, mask, return_aux=return_aux)
