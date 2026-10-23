"""Grouped cross-validation, soft-label training and same-fold self-distillation."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .checkpoints import load_model, save_checkpoint
from .constants import UID, LABELS
from .data import Studies
from .inference import predict_windows, autocast, publish_predictions
from .metrics import auc_metrics
from .model import KneeClassifier
from .sampling import collate


def read_labels(path):
    frame = pd.read_csv(path, dtype={UID: str, 'group': str}).set_index(UID)
    if not frame.index.is_unique or frame[['group','fold']].isna().any().any():
        raise ValueError('Every study needs a unique UID and a group/fold assignment')
    if frame.groupby('group').fold.nunique().max() != 1:
        raise ValueError('A patient/report group crosses validation folds')
    if 'is_gold' in frame and frame.is_gold.fillna(0).astype(bool).any():
        raise ValueError('Keep official gold cases in a separate evaluation table')
    q = frame[LABELS].to_numpy(float)
    if ((q[np.isfinite(q)] < 0) | (q[np.isfinite(q)] > 1)).any():
        raise ValueError('Targets must be probabilities in [0, 1], or missing')
    return frame


def blend_teacher(train, teacher_path, teacher_weight):
    q = train[LABELS].to_numpy(np.float32)
    if teacher_path:
        teacher = pd.read_csv(teacher_path, dtype={UID: str}).set_index(UID)
        if not teacher.index.is_unique or set(teacher.index) != set(train.index):
            raise ValueError('Teacher predictions must cover exactly this fold\'s training partition')
        t = teacher.loc[train.index, LABELS].to_numpy(np.float32)
        if not np.isfinite(t).all() or ((t<0)|(t>1)).any():
            raise ValueError('Teacher probabilities must be finite and in [0,1]')
        q = np.where(np.isfinite(q), (1-teacher_weight)*q + teacher_weight*t, t)
    return q


def train(config_path, labels_path, cache, output, fold, device='cuda:0',
          initial=None, teacher=None, teacher_weight=.25, resume=False, bias_labels=None):
    cfg = json.loads(Path(config_path).read_text())
    frame = read_labels(labels_path)
    training, validation = frame[frame.fold != fold], frame[frame.fold == fold]
    if not len(training) or not len(validation): raise ValueError('Empty training or validation partition')
    target = blend_teacher(training, teacher, teacher_weight)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    seed = cfg['seed'] + fold
    torch.set_num_threads(cfg.get('cpu_threads',4)); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    cfg = {**cfg, 'fold': fold, 'labels': str(Path(labels_path).resolve()), 'initial': initial,
           'teacher': teacher, 'teacher_weight': teacher_weight if teacher else 0.}
    aug = torch.Generator().manual_seed(seed+5000)
    offset = 0
    if initial:
        model = load_model(initial, cfg['crop'], device)
        state = torch.load(initial, weights_only=True, map_location='cpu')
        offset = state.get('epoch', 0)
    else:
        backbone = 'convnext_small.fb_in22k_ft_in1k'
        model = KneeClassifier(backbone, 'timm' if cfg.get('pretrained',True) else None,
            crop=cfg['crop'], spatial=True, extent=True).to(device).to(memory_format=torch.channels_last)
        bias_frame = read_labels(bias_labels).loc[training.index] if bias_labels else training
        means = np.nanmean(bias_frame[LABELS].to_numpy(float), axis=0)
        means = np.nan_to_num(means, nan=.5)
        with torch.no_grad(): model.bias.copy_(torch.logit(torch.tensor(means, device=device).clamp(.02,.98)))
    if cfg.get('gradient_checkpointing',True): model.enc.set_grad_checkpointing(True)
    optimizer = torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if n.startswith('enc.')], 'lr':cfg['lr']},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('enc.')], 'lr':cfg['head_lr']}],
        weight_decay=cfg['weight_decay'])
    accumulation = cfg['accumulation']
    steps_per_epoch = math.ceil(len(training)/accumulation)
    total_steps = steps_per_epoch * cfg['epochs']; warm = max(1,int(total_steps*cfg['warmup_fraction']))
    def factor(step):
        return (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,total_steps-warm)))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    epoch0, best, anchor, stale, history = 0, -1., -1., 0, []
    if resume:
        saved = torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        if saved['config'] != cfg: raise ValueError('Resume requires the original configuration')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['schedule']); torch.set_rng_state(saved['cpu_rng'])
        if str(device).startswith('cuda'): torch.cuda.set_rng_state(saved['cuda_rng'], device)
        aug.set_state(saved['aug_rng'])
        epoch0, best, anchor, stale, history = (saved[k] for k in ['epoch','best','anchor','stale','history'])
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    targets = torch.from_numpy(target).to(device)
    val_ds = Studies(cache, validation.index)
    for epoch in range(epoch0+1, cfg['epochs']+1):
        model.train()
        sampling_epoch = offset+epoch
        ds = Studies(cache,training.index,True,seed,sampling_epoch,cfg['train_windows'])
        order = np.random.default_rng(seed+1000003*sampling_epoch).permutation(len(ds)).tolist()
        loader = torch.utils.data.DataLoader(ds,batch_size=1,sampler=order,num_workers=cfg.get('workers',3),
            collate_fn=collate,pin_memory=str(device).startswith('cuda'),
            prefetch_factor=1 if cfg.get('workers',3) else None,
            generator=torch.Generator().manual_seed(seed+sampling_epoch))
        optimizer.zero_grad(set_to_none=True); losses=[]
        for step,(images,metadata,mask,indices) in enumerate(loader):
            c=cfg['crop']; margin=images.shape[-1]-c
            if margin<0: raise ValueError('Training crop exceeds cached size')
            y,x=torch.randint(margin+1,(2,),generator=aug).tolist()
            images=images[...,y:y+c,x:x+c].to(device)
            metadata,mask=metadata.to(device),mask.to(device)
            q=targets[indices.to(device)]; valid=torch.isfinite(q)
            group_size=min(accumulation,len(loader)-(step//accumulation)*accumulation)
            with autocast(device,'bf16'):
                z=model(images,metadata,mask)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),q.nan_to_num(),reduction='none')
                loss=(loss*valid).sum()/valid.sum().clamp_min(1)
            (loss/group_size).backward();losses.append(float(loss.detach()))
            if (step+1)%accumulation==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True)
        model.eval(); predictions=[]
        for i in range(len(val_ds)):
            im,meta,_=val_ds[i]
            z=predict_windows(model,im,meta,device,cfg['crop'],cfg['predict_chunk'],'bf16')
            predictions.append(torch.sigmoid(torch.from_numpy(z)).numpy())
        predictions=np.stack(predictions)
        metrics=auc_metrics(validation[LABELS].values,predictions)
        score=metrics['macro_auc']
        if score is None: raise ValueError('No validation target has both positive and negative examples')
        record={'epoch':epoch,'loss':float(np.mean(losses)),**metrics}
        history.append(record); print(json.dumps(record),flush=True)
        model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        if score>best:
            best=score
            save_checkpoint(out/'best.pt',{'model':model_state,'epoch':epoch,'config':cfg,'metrics':metrics})
            publish_predictions(out/'best_validation.csv',list(validation.index),predictions)
        if score>anchor+cfg.get('min_delta',.0005): anchor,stale=score,0
        else: stale+=1
        save_checkpoint(out/'last.pt',{'model':model_state,'epoch':epoch,'config':cfg,'metrics':metrics})
        save_checkpoint(out/'resume.pt',{'model':model_state,'config':cfg,'epoch':epoch,'best':best,'anchor':anchor,
            'stale':stale,'history':history,'optimizer':optimizer.state_dict(),'schedule':schedule.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
            'aug_rng':aug.get_state()})
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if cfg.get('patience',0) and epoch>=cfg.get('minimum_epochs',8) and stale>=cfg['patience']:
            break
