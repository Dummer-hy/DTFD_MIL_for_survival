import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lifelines.utils import concordance_index
import pickle
import json
from tqdm import tqdm
import warnings
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)
torch.multiprocessing.set_sharing_strategy('file_system')


class SurvivalWSIDataset(Dataset):
    def __init__(self, pt_folder, csv_file, sample_names):
        self.pt_folder = pt_folder
        self.sample_names = sample_names

        # Load CSV file
        self.df = pd.read_csv(csv_file)

        # Create mapping from 12-character sample prefix to full file name and survival data
        self.prefix_to_data = {}
        for _, row in self.df.iterrows():
            full_name = row['File Name']  # e.g., TCGA-BF-A1PU-01Z-00-DX1.CB0A52E3-16A9-46B2-BBE1-149A6CAAB9CF.svs
            prefix = full_name[:12]  # e.g., TCGA-BF-A1PU

            self.prefix_to_data[prefix] = {
                'full_name': full_name.replace('.svs', '.pt'),  # Convert to .pt extension
                'time': float(row['time']),  # Time to event or last follow-up
                'event': int(row['event'])  # 1=death, 0=censored (alive at last follow-up)
            }

        # Filter sample_names to only include those with survival data
        self.valid_samples = []
        for sample_prefix in sample_names:
            if sample_prefix in self.prefix_to_data:
                self.valid_samples.append(sample_prefix)

        print(f"Dataset initialized with {len(self.valid_samples)} valid samples out of {len(sample_names)} requested")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_prefix = self.valid_samples[idx]  # e.g., TCGA-BF-A1PU

        # Get full filename and survival data
        data = self.prefix_to_data[sample_prefix]
        full_pt_name = data['full_name']  # e.g., TCGA-BF-A1PU-01Z-00-DX1.CB0A52E3-16A9-46B2-BBE1-149A6CAAB9CF.pt

        # Load features
        pt_file = os.path.join(self.pt_folder, full_pt_name)

        try:
            features = torch.load(pt_file)
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            # Ensure features are 2D [num_patches, feature_dim]
            if len(features.shape) == 1:
                features = features.unsqueeze(0)

        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            # Return dummy data if file can't be loaded
            features = torch.randn(100, 1024)

        # Get survival data
        time = torch.tensor(data['time'], dtype=torch.float32)
        event = torch.tensor(data['event'], dtype=torch.float32)

        return {
            'features': features,
            'time': time,
            'event': event,
            'sample_name': sample_prefix,
            'full_name': full_pt_name
        }


# 导入您的模型组件
from Model.network import Attention_Gated as Attention
from Model.network import DimReduction
from utils import get_cam_1d


class SurvivalClassifier(nn.Module):
    """生存分析分类器 - 支持分类和回归模式"""

    def __init__(self, n_channels, droprate=0.0, mode='classification'):
        super(SurvivalClassifier, self).__init__()
        self.mode = mode  # 'classification' 或 'regression'

        if self.mode == 'classification':
            self.classifier = nn.Linear(n_channels, 2)  # 二分类：事件发生/不发生
        else:  # regression
            self.risk_fc = nn.Linear(n_channels, 1)  # 风险评分

        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)

        if self.mode == 'classification':
            logits = self.classifier(x)
            return {'logits': logits}
        else:  # regression
            risk_score = self.risk_fc(x)
            return {'risk_score': risk_score}


# 修复后的Attention_with_SurvivalClassifier类
class Attention_with_SurvivalClassifier(nn.Module):
    """带有生存分析分类器的注意力模块 - 改进版，第二层使用回归模式"""

    def __init__(self, L=512, D=128, K=1, droprate=0):
        super(Attention_with_SurvivalClassifier, self).__init__()
        # 注意力层
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(D, K)

        # 生存分析分类器 - 第二层使用回归模式
        self.survival_classifier = SurvivalClassifier(L, droprate, mode='regression')

    def forward(self, x):  ## x: N x L
        # 计算注意力权重
        A_V = self.attention_V(x)  # N x D
        A_U = self.attention_U(x)  # N x D
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        A = F.softmax(A, dim=1)  # softmax over N

        # 应用注意力权重
        M = torch.mm(A, x)  # K x L

        # 生存分析预测 - 第二层返回risk_score
        survival_pred = self.survival_classifier(M.squeeze(0))
        return survival_pred



class CoxLoss(nn.Module):
    """Cox比例风险模型损失函数"""

    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, risk_scores, events, times):
        """
        Cox比例风险模型的负对数部分似然
        Args:
            risk_scores: 风险评分 tensor [batch_size] 或 [batch_size, 1]
            events: 事件指示器 tensor [batch_size]
            times: 生存时间 tensor [batch_size]
        """
        # 确保所有输入都是1D张量
        if risk_scores.dim() == 0:
            risk_scores = risk_scores.unsqueeze(0)
        elif risk_scores.dim() == 2:
            risk_scores = risk_scores.squeeze(-1)

        if events.dim() == 0:
            events = events.unsqueeze(0)
        elif events.dim() == 2:
            events = events.squeeze(-1)

        if times.dim() == 0:
            times = times.unsqueeze(0)
        elif times.dim() == 2:
            times = times.squeeze(-1)

        # 检查batch大小
        batch_size = risk_scores.shape[0]
        if batch_size == 1:
            # 对于单样本，返回风险评分作为损失（简化版）
            return torch.abs(risk_scores[0])

        # 按时间排序 (降序)
        sorted_indices = torch.argsort(times, descending=True)
        sorted_risk_scores = risk_scores[sorted_indices]
        sorted_events = events[sorted_indices]

        # 计算风险
        risk = torch.exp(sorted_risk_scores)

        # 计算累积风险
        cumsum_risk = torch.cumsum(risk, dim=0)

        # 计算对数部分似然
        event_mask = sorted_events == 1
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

        log_likelihood = sorted_risk_scores[event_mask] - torch.log(cumsum_risk[event_mask] + 1e-8)

        return -torch.mean(log_likelihood)


def calculate_cindex(risk_scores, events, times):
    """计算C-index"""
    try:
        if len(np.unique(events.cpu().numpy())) < 2:
            return 0.5
        cindex = concordance_index(times.cpu().numpy(), -risk_scores.cpu().numpy(), events.cpu().numpy())
        return cindex
    except:
        return 0.5


def calculate_time_dependent_auc(risk_scores, events, times, time_point):
    """计算tAUC"""
    try:
        risk_scores = risk_scores.cpu().numpy()
        events = events.cpu().numpy()
        times = times.cpu().numpy()

        labels = np.zeros(len(times))
        for i in range(len(times)):
            if events[i] == 1 and times[i] <= time_point:
                labels[i] = 1
            elif times[i] > time_point:
                labels[i] = 0
            else:
                labels[i] = -1

        valid_mask = labels != -1
        if valid_mask.sum() < 2 or len(np.unique(labels[valid_mask])) < 2:
            return 0.5

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels[valid_mask], risk_scores[valid_mask])
        return auc
    except:
        return 0.5


def load_splits_from_csv(split_dir):
    """从CSV文件加载数据分割"""
    splits = []

    # 首先检查目录中的文件
    csv_files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]
    print(f"Found CSV files in {split_dir}: {csv_files}")

    for fold in range(5):
        split_file = os.path.join(split_dir, f'splits_{fold}.csv')
        if os.path.exists(split_file):
            split_df = pd.read_csv(split_file)
            # 检查列名并获取样本
            if 'train' in split_df.columns and 'val' in split_df.columns:
                # 方式1: 如果train和val是布尔列或0/1列
                if split_df['train'].dtype == bool or set(split_df['train'].unique()).issubset({0, 1, True, False}):
                    train_samples = split_df[
                        split_df['train'] == 1].index.tolist() if 'sample' not in split_df.columns else \
                    split_df[split_df['train'] == 1]['sample'].tolist()
                    val_samples = split_df[split_df['val'] == 1].index.tolist() if 'sample' not in split_df.columns else \
                    split_df[split_df['val'] == 1]['sample'].tolist()
                else:
                    # 方式2: 如果train和val列包含样本名称
                    train_samples = split_df['train'].dropna().tolist()
                    val_samples = split_df['val'].dropna().tolist()
            else:
                # 检查是否有其他可能的列名
                print(f"Available columns: {split_df.columns.tolist()}")
                # 尝试根据第一列作为样本名，其他列作为标识
                if len(split_df.columns) >= 2:
                    sample_col = split_df.columns[0]  # 假设第一列是样本名

                    # 如果有明确的split类型列
                    if 'split_type' in split_df.columns or 'type' in split_df.columns:
                        split_type_col = 'split_type' if 'split_type' in split_df.columns else 'type'
                        train_samples = split_df[split_df[split_type_col] == 'train'][sample_col].tolist()
                        val_samples = split_df[split_df[split_type_col] == 'val'][sample_col].tolist()
                    else:
                        # 如果没有找到明确的分割信息，抛出错误
                        raise ValueError(f"Cannot determine split structure from columns: {split_df.columns.tolist()}")

            test_samples = val_samples.copy()  # 使用验证集作为测试集

            splits.append({
                'train': train_samples,
                'val': val_samples,
                'test': test_samples
            })
            print(f"Fold {fold}: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        else:
            print(f"Warning: Split file {split_file} not found!")

    return splits

def organize_survival_data(dataset):
    """将数据组织成适合DTFD训练的格式"""
    SlideNames = []
    FeatList = []
    Times = []
    Events = []

    for i in range(len(dataset)):
        data = dataset[i]
        SlideNames.append(data['sample_name'])
        FeatList.append(data['features'])
        Times.append(data['time'].item())
        Events.append(data['event'].item())

    return SlideNames, FeatList, Times, Events


def train_survival_dtfd(mDATA_list, dimReduction, attention, survival_classifier, UClassifier,
                        optimizer0, optimizer1, epoch, cox_criterion=None, ce_criterion=None, params=None,
                        f_log=None, writer=None, numGroup=3, total_instance=3, distill='AFS'):
    """基于DTFD框架的生存分析训练 - 第一层分类，第二层回归"""
    SlideNames_list, mFeat_list, times_list, events_list = mDATA_list

    dimReduction.train()
    attention.train()
    survival_classifier.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    np.random.shuffle(tIDX)  # 使用numpy的shuffle保证固定种子

    for idx in range(numIter):
        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        for slide_idx in tidx_slide:
            tslide_name = SlideNames_list[slide_idx]
            ttime = torch.tensor([times_list[slide_idx]], dtype=torch.float32).to(params.device)
            tevent = torch.tensor([events_list[slide_idx]], dtype=torch.long).to(params.device)  # 改为long类型用于分类

            # 分别为两个阶段准备数据
            slide_pseudo_feat_list = []
            slide_sub_logits = []
            slide_sub_events = []

            tfeat_tensor = mFeat_list[slide_idx]
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_index = list(range(tfeat_tensor.shape[0]))
            np.random.shuffle(feat_index)  # 使用numpy的shuffle
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_events.append(tevent.clone())

                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0,
                                                    index=torch.LongTensor(tindex).to(params.device))
                # 第一阶段：计算注意力和聚合特征
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                # 生存分析分类预测 - 用于第一层损失
                tPredict = survival_classifier(tattFeat_tensor)
                slide_sub_logits.append(tPredict['logits'].clone())

                # 为第二阶段准备特征
                with torch.no_grad():
                    # 获取patch级别的分类logits用于实例选择
                    try:
                        patch_logits = get_cam_1d(survival_classifier.classifier, tattFeats.unsqueeze(0)).squeeze(0)
                        # 使用事件发生类别的概率进行排序
                        patch_scores = F.softmax(patch_logits, dim=-1)[:, 1]  # 取事件发生的概率
                    except:
                        # 如果get_cam_1d失败，使用简单的方法
                        patch_scores = torch.sum(tattFeats * survival_classifier.classifier.weight[1].squeeze(), dim=1)

                    _, sort_idx = torch.sort(patch_scores, descending=True)

                    if distill == 'MaxMinS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        selected_feat = tmidFeat.index_select(dim=0, index=topk_idx).clone()
                        slide_pseudo_feat_list.append(selected_feat)
                    elif distill == 'MaxS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        selected_feat = tmidFeat.index_select(dim=0, index=topk_idx_max).clone()
                        slide_pseudo_feat_list.append(selected_feat)
                    elif distill == 'AFS':
                        slide_pseudo_feat_list.append(tattFeat_tensor.clone())

            # 第一层优化 - 分类损失
            slide_sub_logits = torch.cat(slide_sub_logits, dim=0)
            slide_sub_events = torch.cat(slide_sub_events, dim=0)

            loss0 = ce_criterion(slide_sub_logits, slide_sub_events.squeeze())

            optimizer0.zero_grad()
            loss0.backward()
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(survival_classifier.parameters(), params.grad_clipping)
            optimizer0.step()

            # 第二层优化 - Cox回归损失
            slide_pseudo_feat = torch.cat(slide_pseudo_feat_list, dim=0)
            slide_pseudo_feat = slide_pseudo_feat.detach().requires_grad_(True)

            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = cox_criterion(gSlidePred['risk_score'].squeeze(),
                                  tevent.float().squeeze(),  # 转回float用于Cox损失
                                  ttime.squeeze())

            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = f'epoch: {epoch} idx: {idx}'
            tstr += f' Classification Loss: {Train_Loss0.avg:.4f}, Cox Regression Loss: {Train_Loss1.avg:.4f}'
            print_log(tstr, f_log)

    writer.add_scalar('train_classification_loss', Train_Loss0.avg, epoch)
    writer.add_scalar('train_cox_loss', Train_Loss1.avg, epoch)


def test_survival_dtfd(mDATA_list, dimReduction, attention, survival_classifier, UClassifier, epoch,
                       cox_criterion=None, ce_criterion=None, params=None, f_log=None, writer=None,
                       numGroup=3, total_instance=3, distill='AFS'):
    """基于DTFD框架的生存分析验证 - 第一层分类，第二层回归"""
    dimReduction.eval()
    attention.eval()
    survival_classifier.eval()
    UClassifier.eval()

    SlideNames, FeatLists, times_list, events_list = mDATA_list
    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    # 第一层分类指标
    all_logits_0 = []
    all_events_0 = []

    # 第二层回归指标
    all_risk_scores_1 = []
    all_times = []
    all_events = []

    with torch.no_grad():
        numSlides = len(SlideNames)

        for slide_idx in range(numSlides):
            tslide_name = SlideNames[slide_idx]
            ttime = torch.tensor([times_list[slide_idx]], dtype=torch.float32).to(params.device)
            tevent_float = torch.tensor([events_list[slide_idx]], dtype=torch.float32).to(params.device)
            tevent_long = torch.tensor([events_list[slide_idx]], dtype=torch.long).to(params.device)
            tfeat = FeatLists[slide_idx].to(params.device)

            midFeat = dimReduction(tfeat)
            AA = attention(midFeat, isNorm=False).squeeze(0)

            allSlide_pred_risk = []

            for jj in range(params.num_MeanInference):
                feat_index = list(range(tfeat.shape[0]))
                np.random.shuffle(feat_index)  # 使用numpy的shuffle
                index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                slide_d_feat = []
                slide_sub_logits = []

                for tindex in index_chunk_list:
                    idx_tensor = torch.LongTensor(tindex).to(params.device)
                    tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                    tAA = AA.index_select(dim=0, index=idx_tensor)
                    tAA = torch.softmax(tAA, dim=0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)

                    tPredict = survival_classifier(tattFeat_tensor)
                    slide_sub_logits.append(tPredict['logits'])

                    # 实例选择
                    try:
                        patch_logits = get_cam_1d(survival_classifier.classifier, tattFeats.unsqueeze(0)).squeeze(0)
                        patch_scores = F.softmax(patch_logits, dim=-1)[:, 1]
                    except:
                        patch_scores = torch.sum(tattFeats * survival_classifier.classifier.weight[1].squeeze(), dim=1)

                    _, sort_idx = torch.sort(patch_scores, descending=True)

                    if distill == 'MaxMinS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'MaxS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'AFS':
                        slide_d_feat.append(tattFeat_tensor)

                slide_d_feat = torch.cat(slide_d_feat, dim=0)
                slide_sub_logits = torch.cat(slide_sub_logits, dim=0)

                # 第一层损失 - 分类
                batch_events = tevent_long.repeat(numGroup)
                loss0 = ce_criterion(slide_sub_logits, batch_events.squeeze())
                test_loss0.update(loss0.item(), numGroup)

                # 收集第一层分类结果
                all_logits_0.append(slide_sub_logits.cpu())
                all_events_0.extend(batch_events.cpu().numpy())

                # 第二层预测
                gSlideRisk = UClassifier(slide_d_feat)
                allSlide_pred_risk.append(gSlideRisk['risk_score'])

            # 平均多次推理结果
            allSlide_pred_risk = torch.cat(allSlide_pred_risk, dim=0)
            mean_risk = torch.mean(allSlide_pred_risk, dim=0)

            # 第二层损失
            loss1 = cox_criterion(mean_risk.squeeze(), tevent_float.squeeze(), ttime.squeeze())
            test_loss1.update(loss1.item(), 1)

            # 收集第二层回归结果
            all_risk_scores_1.append(mean_risk.cpu().item())
            all_times.append(ttime.cpu().item())
            all_events.append(tevent_float.cpu().item())

    # 计算第一层分类指标
    all_logits_0 = torch.cat(all_logits_0, dim=0)
    all_events_0 = torch.tensor(all_events_0)
    all_preds_0 = torch.argmax(all_logits_0, dim=1)

    classification_acc = (all_preds_0 == all_events_0).float().mean().item()

    # 计算第二层回归指标
    all_risk_scores_1 = torch.tensor(all_risk_scores_1)
    all_times = torch.tensor(all_times)
    all_events = torch.tensor(all_events)

    cindex_1 = calculate_cindex(all_risk_scores_1, all_events, all_times)

    # 计算tAUC
    median_time = torch.median(all_times[all_events == 1]).item() if (all_events == 1).sum() > 0 else torch.median(
        all_times).item()
    tauc_1 = calculate_time_dependent_auc(all_risk_scores_1, all_events, all_times, median_time)

    print_log(f'  First-Tier Classification Accuracy: {classification_acc:.4f}', f_log)
    print_log(f'  Second-Tier C-index: {cindex_1:.4f}, tAUC: {tauc_1:.4f}', f_log)

    writer.add_scalar('classification_acc', classification_acc, epoch)
    writer.add_scalar('cindex_1', cindex_1, epoch)
    writer.add_scalar('tauc_1', tauc_1, epoch)

    return cindex_1, tauc_1

class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    f.write('\n')
    f.write(tstr)
    f.flush()
    print(tstr)


def main():
    # 固定随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # 训练参数
    class Params:
        def __init__(self):
            self.name = 'TCGA_SKCM_Survival_Classification'
            self.EPOCH = 10
            self.epoch_step = [10]
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.isPar = False
            self.log_dir = './survival_log'
            self.train_show_freq = 10
            self.droprate = 0.5
            self.droprate_2 = 0.5
            self.lr = 1e-4
            self.weight_decay = 1e-4
            self.lr_decay_ratio = 0.2
            self.batch_size = 1
            self.batch_size_v = 1
            self.num_workers = 4
            self.numGroup = 4
            self.total_instance = 4
            self.numGroup_test = 4
            self.total_instance_test = 4
            self.mDim = 512
            self.grad_clipping = 5
            self.isSaveModel = True
            self.numLayer_Res = 0
            self.temperature = 1
            self.num_MeanInference = 1
            self.distill_type = 'AFS'

    params = Params()

    # 数据路径
    # pt_folder = 'D:/Documents/Pycharm/MIL/PORPOISE/datasets/TCGA-SKCM/TCGA-SKCM-pt_files'
    # csv_file = 'D:/Documents/Pycharm/MIL/PORPOISE/datasets/TCGA-SKCM/TCGA-SKCM.csv'
    # split_dir = 'D:/Documents/Pycharm/MIL/PORPOISE/splits/skcm'
    pt_folder = 'D:/Documents/Pycharm/MIL/PORPOISE/datasets/TCGA-HNSC/TCGA-HNSC-pt_files'
    csv_file = 'D:/Documents/Pycharm/MIL/PORPOISE/datasets/TCGA-HNSC/TCGA-HNSC.csv'
    split_dir = 'D:/Documents/Pycharm/MIL/PORPOISE/splits/hnsc'
    print(f"Using device: {params.device}")

    # 加载数据分割
    splits = load_splits_from_csv(split_dir)
    if len(splits) != 5:
        raise ValueError(f"Expected 5 splits, but found {len(splits)}")

    # 五折交叉验证
    fold_results = []

    for fold in range(5):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'=' * 60}")

        # 创建日志
        fold_log_dir = os.path.join(params.log_dir, f'fold_{fold}')
        os.makedirs(fold_log_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(fold_log_dir, 'LOG', params.name))
        log_file = open(os.path.join(fold_log_dir, 'log.txt'), 'w')
        save_dir = os.path.join(fold_log_dir, 'best_model.pth')

        # 获取当前折的样本 - 只使用训练集和验证集
        train_samples = splits[fold]['train']
        val_samples = splits[fold]['val']

        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")

        # 创建数据集
        train_dataset = SurvivalWSIDataset(pt_folder, csv_file, train_samples)
        val_dataset = SurvivalWSIDataset(pt_folder, csv_file, val_samples)

        # 组织数据
        train_data = organize_survival_data(train_dataset)
        val_data = organize_survival_data(val_dataset)

        print_log(f'Training slides: {len(train_data[0])}, '
                  f'Validation slides: {len(val_data[0])}', log_file)

        # 模型设置
        in_chn = 1024

        # 第一层：分类器（事件发生/不发生）
        survival_classifier = SurvivalClassifier(params.mDim, params.droprate, mode='classification').to(params.device)
        attention = Attention(params.mDim).to(params.device)
        dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)

        # 第二层：回归器（风险评分）
        attCls = Attention_with_SurvivalClassifier(L=params.mDim, droprate=params.droprate_2).to(params.device)

        if params.isPar:
            survival_classifier = torch.nn.DataParallel(survival_classifier)
            attention = torch.nn.DataParallel(attention)
            dimReduction = torch.nn.DataParallel(dimReduction)
            attCls = torch.nn.DataParallel(attCls)

        # 损失函数
        cox_criterion = CoxLoss().to(params.device)  # 第二层Cox回归损失
        ce_criterion = nn.CrossEntropyLoss().to(params.device)  # 第一层分类损失

        # 优化器
        trainable_parameters = []
        trainable_parameters += list(survival_classifier.parameters())
        trainable_parameters += list(attention.parameters())
        trainable_parameters += list(dimReduction.parameters())

        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr, weight_decay=params.weight_decay)
        optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, params.epoch_step,
                                                          gamma=params.lr_decay_ratio)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, params.epoch_step,
                                                          gamma=params.lr_decay_ratio)

        best_cindex = 0
        best_epoch = -1
        val_cindex = 0
        val_tauc = 0

        # 训练循环
        for epoch in range(params.EPOCH):

            for param_group in optimizer_adam1.param_groups:
                curLR = param_group['lr']
                print_log(f'Current learning rate: {curLR}', log_file)

            # 训练
            train_survival_dtfd(dimReduction=dimReduction, attention=attention,
                                survival_classifier=survival_classifier, UClassifier=attCls,
                                mDATA_list=train_data, cox_criterion=cox_criterion, ce_criterion=ce_criterion,
                                optimizer0=optimizer_adam0, optimizer1=optimizer_adam1,
                                epoch=epoch, params=params, f_log=log_file, writer=writer,
                                numGroup=params.numGroup, total_instance=params.total_instance,
                                distill=params.distill_type)

            # 验证
            print_log(f'>>>>>>>>>>> Validation Epoch: {epoch}', log_file)
            val_cindex_curr, val_tauc_curr = test_survival_dtfd(dimReduction=dimReduction, attention=attention,
                                                                survival_classifier=survival_classifier,
                                                                UClassifier=attCls,
                                                                mDATA_list=val_data, cox_criterion=cox_criterion,
                                                                ce_criterion=ce_criterion,
                                                                epoch=epoch, params=params, f_log=log_file,
                                                                writer=writer,
                                                                numGroup=params.numGroup_test,
                                                                total_instance=params.total_instance_test,
                                                                distill=params.distill_type)

            # 保存最佳模型
            if epoch > int(params.EPOCH * 0.8):
                if val_cindex_curr > best_cindex:
                    best_cindex = val_cindex_curr
                    best_epoch = epoch
                    val_cindex = val_cindex_curr
                    val_tauc = val_tauc_curr

                    if params.isSaveModel:
                        tsave_dict = {
                            'survival_classifier': survival_classifier.state_dict(),
                            'dim_reduction': dimReduction.state_dict(),
                            'attention': attention.state_dict(),
                            'att_classifier': attCls.state_dict(),
                            'fold': fold,
                            'epoch': epoch,
                            'val_cindex': val_cindex,
                            'val_tauc': val_tauc
                        }
                        torch.save(tsave_dict, save_dir)

                print_log(f'Best Val C-index: {val_cindex:.4f}, Val tAUC: {val_tauc:.4f}, from epoch {best_epoch}',
                          log_file)

            scheduler0.step()
            scheduler1.step()

        # 记录当前折结果
        fold_results.append({
            'fold': fold + 1,
            'best_val_cindex': best_cindex,
            'val_cindex': val_cindex,
            'val_tauc': val_tauc,
            'best_epoch': best_epoch
        })

        print_log(f'\nFold {fold + 1} Final Results:', log_file)
        print_log(f'Best Val C-index: {best_cindex:.4f}', log_file)
        print_log(f'Val tAUC: {val_tauc:.4f}', log_file)

        log_file.close()
        writer.close()

    # 计算交叉验证结果
    val_cindexes = [result['val_cindex'] for result in fold_results]
    val_taucs = [result['val_tauc'] for result in fold_results]

    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean Val C-index: {np.mean(val_cindexes):.4f} ± {np.std(val_cindexes):.4f}")
    print(f"Mean Val tAUC: {np.mean(val_taucs):.4f} ± {np.std(val_taucs):.4f}")

    # 保存最终结果
    final_results = {
        'fold_results': fold_results,
        'summary': {
            'mean_val_cindex': float(np.mean(val_cindexes)),
            'std_val_cindex': float(np.std(val_cindexes)),
            'mean_val_tauc': float(np.mean(val_taucs)),
            'std_val_tauc': float(np.std(val_taucs))
        },
        'params': vars(params)
    }

    results_file = os.path.join(params.log_dir, 'cross_validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()