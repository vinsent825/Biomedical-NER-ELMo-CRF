# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 17:34:28 2026

@author: vinsent825
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 

# ==========================================
# 0. Global Configuration
# ==========================================
CONFIG = {
    'char_vocab': 100,
    'word_vocab': 5000,
    'char_dim': 16,
    'word_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'seq_len': 64,
    'char_len': 20,
    'batch_size': 8
}

 
    
    


# ==========================================
# 1. Base Module: Char-CNN (Handles Biomedical Term Variations)
# ==========================================
class CharCNNEmbedding(nn.Module):
    def __init__(self, vocab_size, char_embed_dim, word_embed_dim,
                 kernel_sizes=[3,4,5], num_filters_per_kernel=64, padding_idx=0):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=padding_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters_per_kernel, k, padding=k//2)
            for k in kernel_sizes
        ])
        
        total_filters = num_filters_per_kernel * len(kernel_sizes)
        self.projection = nn.Linear(total_filters, word_embed_dim)   
        
        

    def forward(self, char_ids):
        B, S, C = char_ids.shape
        x = char_ids.view(-1, C)                # (B*S, C)
        x = self.char_embedding(x)              # (B*S, C, E)
        x = x.transpose(1, 2)                   # (B*S, E, C)
        
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)                       # (B*S, F, ~C)
            out = F.relu(out)
            out = F.max_pool1d(out, out.size(2)).squeeze(2)  # (B*S, F)
            conv_outputs.append(out)
            
        x = torch.cat(conv_outputs, dim=1)      # (B*S, total_filters)
        x = self.projection(x)                  # (B*S, word_dim)
        
        return x.view(B, S, -1)

# ==========================================
# 2. Core Module: ELMo BiLM (Pre-training Capability)
# ==========================================
class ELMo_BiLM_Core(nn.Module):
    """
    Core Language Model for ELMo.
    Contains:
    1. Projection Layer: Aligns Char-CNN and LSTM dimensions
    2. Stacked Bi-LSTMs: Independent Forward/Backward networks
    3. Prediction Heads: Classification heads for pre-training
    """
    def __init__(self, char_vocab_size, char_embed_dim, word_embed_dim, 
                 hidden_dim, word_vocab_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim   
        
     
        self.char_cnn = CharCNNEmbedding(char_vocab_size, char_embed_dim, word_embed_dim)
        
        
        
   
        self.embed_projection = nn.Linear(word_embed_dim, hidden_dim * 2)
        
     
        self.fwd_lstms = nn.ModuleList()
        self.bwd_lstms = nn.ModuleList()
        
        for i in range(num_layers):           
            input_size = hidden_dim * 2 
            self.fwd_lstms.append(nn.LSTM(input_size, hidden_dim, batch_first=True))
            self.bwd_lstms.append(nn.LSTM(input_size, hidden_dim, batch_first=True))
       
        self.fwd_classifier = nn.Linear(hidden_dim, word_vocab_size)
        self.bwd_classifier = nn.Linear(hidden_dim, word_vocab_size)
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(p=0.1)  
        


    def forward(self, char_ids):
       
        token_embeddings = self.char_cnn(char_ids) 
        
       
        projected_embeddings = self.embed_projection(token_embeddings) # (B, Seq, Hidden*2)
        all_layers_outputs = [projected_embeddings]
        
       
        fwd_input = projected_embeddings
        bwd_input = torch.flip(projected_embeddings, [1]) 
        
        final_fwd_state = None
        final_bwd_state = None
        
       
        for i in range(self.num_layers):
            fwd_out, _ = self.fwd_lstms[i](fwd_input)
            bwd_out_reversed, _ = self.bwd_lstms[i](bwd_input)
            bwd_out = torch.flip(bwd_out_reversed, [1])
            
            if i == self.num_layers - 1:
                final_fwd_state = fwd_out
                final_bwd_state = bwd_out
            
            layer_output = torch.cat([fwd_out, bwd_out], dim=-1)
            layer_output = self.layer_norms[i](layer_output)
            layer_output = self.dropout(layer_output)
            
            all_layers_outputs.append(layer_output)
            
            
            fwd_input = layer_output + fwd_input
            bwd_input = torch.flip(fwd_input, dims=[1])
            
      
        fwd_logits = self.fwd_classifier(final_fwd_state)
        bwd_logits = self.bwd_classifier(final_bwd_state)
        
        return all_layers_outputs, (fwd_logits, bwd_logits)

class ScalarMix(nn.Module):
  
    def __init__(self, num_layers):
        super().__init__()
        self.scalar_parameters = nn.Parameter(torch.zeros(num_layers)) 
        self.gamma = nn.Parameter(torch.ones(1)) 

    def forward(self, layer_outputs):
       
        norm_weights = F.softmax(self.scalar_parameters, dim=0)
        
        weighted_sum = torch.zeros_like(layer_outputs[0])
        for i, output in enumerate(layer_outputs):
            weighted_sum += norm_weights[i] * output
            
        return self.gamma * weighted_sum



class CRF(nn.Module):
  
    def __init__(self, num_tags, pad_idx=None):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx
        
       
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self._init_constraints()
    
    def _init_constraints(self):
       
       
        pass
    
    def _compute_score(self, emissions, tags, mask):
        """計算給定標籤序列的分數"""
        batch_size, seq_len = tags.shape
        
      
        lengths = mask.sum(dim=1).long() 
        
       
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        
       
        for i in range(1, seq_len):
            current_tag = tags[:, i]
            previous_tag = tags[:, i - 1]
            
            emit_score = emissions[:, i].gather(1, current_tag.unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[current_tag, previous_tag]
            
           
            step_score = emit_score + trans_score
            score = torch.where(mask[:, i].bool(), score + step_score, score)
        
       
        last_tag_indices = (lengths - 1).clamp(min=0)
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_normalizer(self, emissions, mask):
       
        batch_size, seq_len, num_tags = emissions.shape
        
        
        score = self.start_transitions + emissions[:, 0]
        
       
        transitions_t = self.transitions.t()  
        
       
        for i in range(1, seq_len):
           
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            
            next_score = broadcast_score + transitions_t + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            
            
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
        
       
        score += self.end_transitions
        
        return torch.logsumexp(score, dim=1)

    
    def forward(self, emissions, tags, mask=None):
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.float)
        
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        
        return (denominator - numerator).mean()
    
    def decode(self, emissions, mask=None):
      
        batch_size, seq_len, num_tags = emissions.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.float, device=emissions.device)
        
       
        lengths = mask.sum(dim=1).long()  # (batch_size,)
        
       
        transitions_t = self.transitions.t()  # (num_tags, num_tags)
        
      
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            # broadcast_score: (batch, num_tags, 1) - prev_tag
            # transitions_t: (num_tags, num_tags) - [prev_tag, curr_tag]
            # broadcast_emissions: (batch, 1, num_tags) - curr_tag
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + transitions_t + broadcast_emissions
            
            
            next_score, indices = next_score.max(dim=1)
            
            
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)
        
        
        score += self.end_transitions
        
      
        best_paths = []
        
        for batch_idx in range(batch_size):
            seq_len_i = lengths[batch_idx].item()
            
           
            best_last_tag = score[batch_idx].argmax().item()
            best_path = [best_last_tag]
            
            
            for hist_idx in range(seq_len_i - 2, -1, -1):
                best_last_tag = history[hist_idx][batch_idx][best_last_tag].item()
                best_path.append(best_last_tag)
            
            best_path.reverse()
            best_paths.append(best_path)
        
        return best_paths
    
    



class BioNER_Model(nn.Module):
    """
    Complete BioNER model:
    Input -> ELMo (frozen) -> ScalarMix -> BiLSTM -> CRF -> Tags
    
    Supports:
    - Training mode: returns CRF loss
    - Inference mode: returns Viterbi decoding results
    """
    def __init__(self, elmo_core, hidden_dim, num_tags, static_embed_dim=0, 
                 freeze_elmo=True, pad_idx=0):
        super().__init__()
        self.elmo_core = elmo_core
        self.freeze_elmo = freeze_elmo
        self.num_tags = num_tags
        self.pad_idx = pad_idx
        
        # Freeze ELMo parameters
        if freeze_elmo:
            for param in self.elmo_core.parameters():
                param.requires_grad = False
        
        # ScalarMix layers = ELMo layers + Embedding layer
        self.scalar_mix = ScalarMix(num_layers=elmo_core.num_layers + 1)
        
        # ELMo output dimension is hidden_dim * 2
        input_dim = (elmo_core.hidden_dim * 2) + static_embed_dim
        
        # NER feature extraction layer
        self.ner_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        
        # Emission score layer (replaces original classifier)
        self.hidden2emit = nn.Linear(hidden_dim * 2, num_tags)
        
        # CRF decoding layer
        self.crf = CRF(num_tags, pad_idx=pad_idx)
    
    def _get_elmo_repr(self, char_ids):
        """Get ELMo representation, freeze based on setting"""
        if self.freeze_elmo:
            with torch.no_grad():
                all_layers, _ = self.elmo_core(char_ids)
        else:
            all_layers, _ = self.elmo_core(char_ids)
        return self.scalar_mix(all_layers)
    
    def _get_emissions(self, char_ids, static_embeddings=None):
        """Compute emission scores"""
        elmo_repr = self._get_elmo_repr(char_ids)
        
        if static_embeddings is not None:
            final_input = torch.cat([elmo_repr, static_embeddings], dim=-1)
        else:
            final_input = elmo_repr
        
        ner_out, _ = self.ner_lstm(final_input)
        ner_out = self.dropout(ner_out)
        emissions = self.hidden2emit(ner_out)
        
        return emissions
    
    def forward(self, char_ids, tags=None, mask=None, static_embeddings=None):
        """
        Unified interface for training and inference
        
        Args:
            char_ids: (batch, seq_len, char_len) character IDs
            tags: (batch, seq_len) true labels, provided during training
            mask: (batch, seq_len) valid position mask
            static_embeddings: optional static word embeddings
        
        Returns:
            Training mode: loss (scalar)
            Inference mode: List[List[int]] predicted tag sequences
        """
        emissions = self._get_emissions(char_ids, static_embeddings)
        
        if tags is not None:
            # Training mode: compute CRF loss
            return self.crf(emissions, tags, mask)
        else:
            # Inference mode: Viterbi decoding
            return self.crf.decode(emissions, mask)




def compute_bi_lm_loss(predictions, targets, ignore_index=0, reduction='mean'):
    """
    Compute bidirectional language modeling loss.
    Forward: Predict t+1 at position t.
    Backward: Predict t-1 at position t.
    
    Args:
        predictions: tuple (fwd_logits, bwd_logits), both (B, L, V)
        targets: word token ids (B, L)
        ignore_index: padding token id
    """
    fwd_logits, bwd_logits = predictions
    vocab_size = fwd_logits.size(-1)
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    
    # Forward LM: predict next token
    # shift right → predict token[t+1] using token[0~t]
    loss_fwd = criterion(
        fwd_logits[:, :-1].reshape(-1, vocab_size),     # (B*(L-1), V)
        targets[:, 1:].reshape(-1)                      # (B*(L-1))
    )
    
    # Backward LM: predict previous token
    # shift left → predict token[t-1] using token[t~end]
    loss_bwd = criterion(
        bwd_logits[:, 1:].reshape(-1, vocab_size),      # (B*(L-1), V)
        targets[:, :-1].reshape(-1)                     # (B*(L-1))
    )
    
    return loss_fwd + loss_bwd






class NERMetrics:
    """
    NER metrics calculator
    Supports Entity-level Precision/Recall/F1
    """
    def __init__(self, tag_vocab):
        self.tag_vocab = tag_vocab
        self.id2tag = {v: k for k, v in tag_vocab.items()}
    
    def _extract_entities(self, tag_ids, mask=None):
        """
        Extract entities from the tag sequence
        Returns: Set[(entity_type, start, end)]
        """
        entities = set()
        seq_len = len(tag_ids)
        
        if mask is not None:
            seq_len = int(sum(mask))
        
        i = 0
        while i < seq_len:
            tag = self.id2tag.get(tag_ids[i], 'O')
            
            if tag.startswith('B-'):
                entity_type = tag[2:]
                start = i
                i += 1
                
                while i < seq_len:
                    next_tag = self.id2tag.get(tag_ids[i], 'O')
                    if next_tag == f'I-{entity_type}':
                        i += 1
                    else:
                        break
                
                entities.add((entity_type, start, i))
            else:
                i += 1
        
        return entities
    
    def compute_batch_metrics(self, pred_sequences, gold_sequences, masks=None):
        """
        Compute batch TP/FP/FN
        
        Args:
            pred_sequences: List[List[int]] predicted tags
            gold_sequences: Tensor (batch, seq_len) true labels
            masks: Tensor (batch, seq_len) optional
        
        Returns:
            dict with tp, fp, fn counts per entity type
        """
        metrics = {'total': {'tp': 0, 'fp': 0, 'fn': 0}}
        
        batch_size = len(pred_sequences)
        
        for i in range(batch_size):
            pred_tags = pred_sequences[i]
            gold_tags = gold_sequences[i].tolist()
            mask = masks[i].tolist() if masks is not None else None
            
            pred_entities = self._extract_entities(pred_tags, mask)
            gold_entities = self._extract_entities(gold_tags, mask)
            
            tp = pred_entities & gold_entities
            fp = pred_entities - gold_entities
            fn = gold_entities - pred_entities
            
            metrics['total']['tp'] += len(tp)
            metrics['total']['fp'] += len(fp)
            metrics['total']['fn'] += len(fn)
            
            # Per-category statistics
            for entity_type in set(e[0] for e in pred_entities | gold_entities):
                if entity_type not in metrics:
                    metrics[entity_type] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                type_pred = {e for e in pred_entities if e[0] == entity_type}
                type_gold = {e for e in gold_entities if e[0] == entity_type}
                
                metrics[entity_type]['tp'] += len(type_pred & type_gold)
                metrics[entity_type]['fp'] += len(type_pred - type_gold)
                metrics[entity_type]['fn'] += len(type_gold - type_pred)
        
        return metrics
    
    @staticmethod
    def compute_f1(tp, fp, fn):
        """Compute Precision/Recall/F1"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def summary(self, metrics):
        """Format and output evaluation results"""
        results = {}
        for key, counts in metrics.items():
            results[key] = self.compute_f1(counts['tp'], counts['fp'], counts['fn'])
        return results

    


 

import nltk
from torch.utils.data import Dataset, DataLoader
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  





virtual_dataset = [
    "Long-term use of NSAIDs was associated with increased gastrointestinal bleeding in a multicenter hypertensive patients. cystic fibrosis showed 80% higher incidence of myocardial infarction. Switching to CRP reduced these risks. imaging findings levels correlated with outcomes.",
    "Long-term use of statins was associated with increased gastrointestinal bleeding in a large hypertensive patients. amplification showed 64% higher incidence of heart failure. Switching to CRP reduced these risks. serum levels levels correlated with outcomes.",
    "Alterations in the gut microbiome were investigated in non-small cell lung cancer patients using 16S rRNA sequencing. reduced diversity was commonly observed. Intervention with fecal microbiota transplantation led to mucosal healing. Longitudinal analysis confirmed diversity index as key to potent anti-tumor effects.",
    "Long-term use of corticosteroids was associated with increased gastrointestinal bleeding in a large diabetic patients. off-target showed 85% higher incidence of stroke. Switching to dysbiosis reduced these risks. inflammatory markers levels correlated with outcomes. Further mechanistic studies are required to elucidate the underlying pathways.",
    "Gene editing using prime editing successfully corrected the ΔF508 in breast cancer models. Restoration of channel activity was achieved in 76% of cells. AAV vectors enabled hepatic targeting. Future work should address off-target effects. Safety profile was generally acceptable with manageable toxicities.",
    "T790M signaling plays a critical role in breast cancer progression. Novel inhibitors such as microbiome exhibited strong inhibition in xenograft models. Gene expression profiling indicated suppression of related pathways. These findings suggest potential for targeted therapy in advanced-stage patients. These results highlight the importance of personalized medicine approaches.",
    "Long-term use of statins was associated with increased renal impairment in 1200 hypertensive patients. ulcerative colitis showed 66% higher incidence of myocardial infarction. Switching to Lactobacillus reduced these risks. serum levels levels correlated with outcomes. These results highlight the importance of personalized medicine approaches.",
    "Crizotinib signaling plays a critical role in pancreatic cancer progression. Novel inhibitors such as CRISPR-Cas9 exhibited strong downregulation in xenograft models. Gene expression profiling indicated suppression of downstream pathways. These findings suggest potential for targeted therapy in advanced-stage patients.",
    "probiotic signaling plays a critical role in colorectal cancer progression. Novel inhibitors such as Gefitinib exhibited strong downregulation in preclinical models. Gene expression profiling indicated suppression of related pathways. These findings suggest potential for targeted therapy in advanced-stage patients. These results highlight the importance of personalized medicine approaches.",
    "Gene editing using base editing successfully corrected the common mutation in pancreatic cancer models. Restoration of channel activity was achieved in 63% of cells. nanoparticle vectors enabled lung targeting. Future work should address immune response. Further mechanistic studies are required to elucidate the underlying pathways.",
    "Long-term use of NSAIDs was associated with increased cardiovascular risk in a multicenter diabetic patients. Naproxen showed 81% higher incidence of stroke. Switching to CFTR reduced these risks. inflammatory markers levels correlated with outcomes. Safety profile was generally acceptable with manageable toxicities.",
    "Gene editing using base editing successfully corrected the ΔF508 in glioblastoma models. Restoration of protein function was achieved in 84% of cells. nanoparticle vectors enabled lung targeting. Future work should address immune response.",
    "Long-term use of statins was associated with increased renal impairment in 1200 elderly patients. CRP showed 64% higher incidence of stroke. Switching to Afatinib reduced these risks. serum levels levels correlated with outcomes. Safety profile was generally acceptable with manageable toxicities.",
    "Alterations in the oral microbiota were investigated in glioblastoma patients using metagenomic sequencing. reduced diversity was commonly observed. Intervention with probiotic supplementation led to mucosal healing. Longitudinal analysis confirmed microbiota composition as key to reduced disease activity. Further mechanistic studies are required to elucidate the underlying pathways.",
    "Long-term use of statins was associated with increased renal impairment in a large hypertensive patients. 16S rRNA showed 76% higher incidence of heart failure. Switching to Lactobacillus reduced these risks. imaging findings levels correlated with outcomes. Safety profile was generally acceptable with manageable toxicities.",
    "Alterations in the gut microbiome were investigated in glioblastoma patients using 16S rRNA sequencing. increased pathogenic taxa was commonly observed. Intervention with fecal microbiota transplantation led to symptom alleviation. Longitudinal analysis confirmed microbiota composition as key to significant tumor regression. Safety profile was generally acceptable with manageable toxicities.",
    "CRP signaling plays a critical role in glioblastoma progression. Novel inhibitors such as glioblastoma exhibited strong anti-tumor activity in xenograft models. Gene expression profiling indicated suppression of target pathways. These findings suggest potential for clinical translation in refractory patients. The findings support the potential for biomarker-driven patient stratification.",
    "The study evaluated the efficacy of CRISPR-Cas9-tinib in patients with CFTR amplification. Results demonstrated reduced disease activity in 84% of cases, however disease progression was observed. Combination with 16S rRNA showed improved tolerability. Adverse events were primarily mild fatigue. Larger trials are warranted.",
    "Long-term use of NSAIDs was associated with increased renal impairment in a multicenter diabetic patients. angiogenesis showed 65% higher incidence of heart failure. Switching to Osimertinib reduced these risks. inflammatory markers levels correlated with outcomes.",
    "ulcerative colitis signaling plays a critical role in non-small cell lung cancer progression. Novel inhibitors such as 16S rRNA exhibited strong anti-tumor activity in cell lines. Gene expression profiling indicated suppression of related pathways. These findings suggest potential for clinical translation in advanced-stage patients.",
    "The study evaluated the efficacy of CFTR-tinib in patients with MET fusion. Results demonstrated significant tumor regression in 80% of cases, however resistance mechanism was observed. Combination with amplification showed improved tolerability. Adverse events were primarily nausea. Larger trials are warranted. Further mechanistic studies are required to elucidate the underlying pathways.",
    "Alterations in the gut microbiome were investigated in colorectal cancer patients using 16S rRNA sequencing. reduced diversity was commonly observed. Intervention with fecal microbiota transplantation led to mucosal healing. Longitudinal analysis confirmed diversity index as key to improved progression-free survival.",
    "The study evaluated the efficacy of NSAID in patients with cystic fibrosis mutation. Results demonstrated reduced disease activity in 63% of cases, however secondary mutation was observed. Combination with cystic fibrosis showed improved overall survival. Adverse events were primarily rash. Larger trials are warranted. The findings support the potential for biomarker-driven patient stratification.",
    "Alterations in the gut microbiome were investigated in pancreatic cancer patients using metagenomic sequencing. increased pathogenic taxa was commonly observed. Intervention with probiotic supplementation led to symptom alleviation. Longitudinal analysis confirmed diversity index as key to reduced disease activity.",
    "Long-term use of corticosteroids was associated with increased gastrointestinal bleeding in a multicenter hypertensive patients. remission showed 90% higher incidence of stroke. Switching to Celecoxib reduced these risks. serum levels levels correlated with outcomes.",
    "Gene editing using prime editing successfully corrected the common mutation in pancreatic cancer models. Restoration of channel activity was achieved in 76% of cells. nanoparticle vectors enabled intestinal targeting. Future work should address off-target effects.",
    "The study evaluated the efficacy of NSAID-zumab in patients with amplification mutation. Results demonstrated significant tumor regression in 62% of cases, however secondary mutation was observed. Combination with Ibuprofen showed improved response rate. Adverse events were primarily nausea. Larger trials are warranted.",
    "The study evaluated the efficacy of CFTR-tinib in patients with ulcerative colitis mutation. Results demonstrated significant tumor regression in 87% of cases, however resistance mechanism was observed. Combination with ulcerative colitis showed improved overall survival. Adverse events were primarily rash. Larger trials are warranted. Safety profile was generally acceptable with manageable toxicities.",
    "Long-term use of NSAIDs was associated with increased cardiovascular risk in a large elderly patients. PT2385 showed 76% higher incidence of stroke. Switching to PDGF reduced these risks. inflammatory markers levels correlated with outcomes. Further mechanistic studies are required to elucidate the underlying pathways.",
    "Long-term use of NSAIDs was associated with increased cardiovascular risk in 500 hypertensive patients. microbiome showed 89% higher incidence of heart failure. Switching to Osimertinib reduced these risks. inflammatory markers levels correlated with outcomes.",
  
    "The efficacy of Aspirin in preventing myocardial infarction was studied in a cohort of diabetic patients. Results showed a 45% reduction in incidence. However, gastrointestinal bleeding was a common side effect. Monitoring CRP levels is recommended.",
    "Patients with ulcerative colitis often experience flare-ups. Treatment with corticosteroids can induce remission, but long-term use increases risk of osteoporosis. Microbiome analysis revealed dysbiosis in affected individuals.",
    "In non-small cell lung cancer, EGFR mutations like T790M are common. Osimertinib has shown superior progression-free survival compared to Gefitinib. Gene sequencing is crucial for personalized therapy.",
    "Cystic fibrosis patients benefit from CFTR modulators such as Ivacaftor. Clinical trials demonstrated improved lung function. However, monitoring for liver enzymes is necessary due to potential hepatotoxicity.",
    "The role of statins in cardiovascular risk reduction is well-established. In hypertensive patients, Atorvastatin reduced stroke incidence by 30%. Serum cholesterol levels were significantly lowered.",
    "Glioblastoma multiforme remains challenging to treat. Temozolomide combined with radiation therapy is standard. Recent studies explore immunotherapy approaches targeting PD-1 pathways.",
    "Breast cancer screening with mammography has reduced mortality. For HER2-positive cases, Trastuzumab improves outcomes. Genetic testing for BRCA mutations guides preventive strategies.",
    "Pancreatic cancer often presents late. Gemcitabine is first-line chemotherapy. Research into KRAS inhibitors shows promise for targeted therapy in mutated cases.",
    "Colorectal cancer prevention includes colonoscopy screening. In advanced stages, Bevacizumab targets angiogenesis. Microbiota alterations may influence disease progression.",
    "Rheumatoid arthritis management involves DMARDs like Methotrexate. Biologics such as Adalimumab target TNF-alpha. Monitoring for infections is essential.",
    "Alzheimer's disease progression involves beta-amyloid plaques. Donepezil improves cognition temporarily. Ongoing trials test anti-amyloid antibodies.",
    "Parkinson's disease symptoms include tremors. Levodopa remains cornerstone treatment. Deep brain stimulation helps in advanced cases.",
    "Multiple sclerosis relapses can be managed with high-dose corticosteroids. Disease-modifying therapies like Fingolimod reduce relapse rates.",
    "Chronic kidney disease in diabetic patients progresses slowly. ACE inhibitors like Enalapril slow progression. Monitoring eGFR is crucial.",
    "Hepatitis C treatment with direct-acting antivirals achieves high cure rates. Sofosbuvir-based regimens are effective across genotypes.",
    "Asthma control involves inhaled corticosteroids. For severe cases, biologics targeting IL-5 like Mepolizumab are used.",
    "Osteoporosis prevention in postmenopausal women includes bisphosphonates. Alendronate reduces fracture risk significantly.",
    "Depression treatment options include SSRIs like Sertraline. Cognitive behavioral therapy complements pharmacotherapy.",
    "Anxiety disorders respond to benzodiazepines short-term. Long-term, SSRIs are preferred to avoid dependence.",
    "Schizophrenia management involves antipsychotics. Olanzapine controls positive symptoms effectively.",
    "HIV antiretroviral therapy suppresses viral load. Tenofovir-based regimens prevent transmission.",
    "Tuberculosis treatment requires multi-drug regimens. Rifampin is a key component.",
    "Malaria prophylaxis in endemic areas uses Atovaquone-Proguanil. Prompt treatment prevents complications.",
    "COVID-19 vaccines reduce severe disease. Paxlovid treats high-risk cases effectively.",
    "Influenza vaccination prevents outbreaks. Oseltamivir shortens symptom duration.",
    "Diabetes management includes Metformin as first-line. Insulin therapy for advanced cases.",
    "Hypertension control with beta-blockers like Metoprolol. Lifestyle modifications enhance efficacy.",
    "Hyperthyroidism treated with Methimazole. Radioiodine for definitive management.",
    "Hypothyroidism requires Levothyroxine replacement. TSH monitoring guides dosing.",
    "Gout flares managed with Colchicine. Allopurinol prevents recurrent attacks.",
    "Psoriasis treatment with topical corticosteroids. Biologics for severe cases.",
    "Eczema managed with emollients. Calcineurin inhibitors for facial areas.",
    "Acne vulgaris responds to benzoyl peroxide. Isotretinoin for severe nodulocystic cases.",
    "Migraine prophylaxis with Propranolol. Triptans abort acute attacks.",
    "Epilepsy control with Valproate. Lamotrigine as alternative in women.",
    "Stroke prevention in atrial fibrillation uses Warfarin. DOACs like Apixaban preferred.",
    "Heart failure management includes ACE inhibitors. Beta-blockers improve ejection fraction.",
    "COPD exacerbations treated with antibiotics. Inhaled bronchodilators for maintenance.",
    "Pneumonia empiric therapy with Amoxicillin-Clavulanate. Viral causes require supportive care.",
    "Sepsis protocol includes broad-spectrum antibiotics. Fluid resuscitation critical.",
    "Cancer pain managed with opioids. Adjuvants like Gabapentin for neuropathic component.",
    "Palliative care in terminal illness focuses on symptom control. Hospice services provide support.",
    "Obesity management involves lifestyle interventions. Bariatric surgery for morbid cases.",
    "Anemia in chronic disease responds to erythropoietin. Iron supplementation if deficient.",
    "Thrombocytopenia in ITP treated with corticosteroids. Rituximab for refractory cases.",
    "Hemophilia A requires factor VIII replacement. Gene therapy emerging.",
    "Sickle cell disease managed with Hydroxyurea. Bone marrow transplant curative in some.",
    "Lymphoma chemotherapy with R-CHOP. Targeted therapies for specific subtypes.",
    "Leukemia treatment varies by type. Imatinib for CML transformed outcomes.",
    "Myeloma managed with proteasome inhibitors like Bortezomib. Stem cell transplant consolidates response.",
    "Alterations in the skin microbiome were investigated in eczema patients using 16S rRNA sequencing. Increased Staphylococcus aureus was observed. Intervention with topical probiotics led to symptom improvement.",
    "Long-term use of antibiotics was associated with gut dysbiosis in a cohort study. Clostridium difficile infection risk increased by 70%. Fecal microbiota transplantation restored balance.",
    "Gene therapy for hemophilia using AAV vectors achieved sustained factor levels. Off-target integration remains a concern. Long-term follow-up is essential.",
    "CRISPR-based editing corrected sickle cell mutation in stem cells. Clinical translation ongoing. Ethical considerations discussed.",
    "Nanoparticle drug delivery enhanced chemotherapy efficacy in pancreatic cancer. Reduced systemic toxicity observed. Targeted ligands improved specificity.",
    "Immunotherapy with CAR-T cells revolutionized lymphoma treatment. Cytokine release syndrome managed with tocilizumab.",
    "Precision medicine in oncology uses next-generation sequencing. Actionable mutations guide therapy selection.",
    "Viral vector vaccines for Ebola demonstrated high efficacy. Ring vaccination strategy contained outbreaks.",
    "mRNA vaccine technology accelerated COVID-19 response. Lipid nanoparticles enabled delivery.",
    "Stem cell therapy for spinal cord injury shows promise. Neuronal regeneration observed in models.",
    "Organoid models recapitulate tumor microenvironment. Drug screening platforms developed.",
    "Liquid biopsy detects circulating tumor DNA. Early detection and monitoring enabled.",
    "AI-assisted diagnosis improves radiology accuracy. Deep learning algorithms detect anomalies.",
    "Wearable devices monitor cardiac arrhythmias. Real-time alerts prevent adverse events.",
    "Telemedicine expands access to specialists. Rural patients benefit significantly.",
    "Blockchain secures health data exchange. Patient privacy preserved.",
    "3D printing customizes prosthetics. Improved fit and function.",
    "Robotic surgery enhances precision. Reduced recovery times.",
    "Augmented reality aids surgical planning. Overlay of imaging data.",
    "Virtual reality therapy for phobias. Exposure in controlled environment.",
    "The impact of NSAIDs on renal function in elderly patients was examined. Chronic use led to a 50% increase in acute kidney injury. Switching to selective COX-2 inhibitors like Celecoxib mitigated risks.",
    "Microbiome modulation in inflammatory bowel disease shows promise. Probiotic strains of Bifidobacterium improved symptoms in Crohn's disease patients.",
    "Targeted therapy for ALK-positive lung cancer with Crizotinib extended survival. Resistance mechanisms involve secondary mutations.",
    "CFTR correctors like Lumacaftor combined with Ivacaftor treat cystic fibrosis. Sweat chloride levels decreased significantly.",
    "Statins such as Simvastatin lower LDL cholesterol. In dyslipidemia, they reduce atherosclerotic plaque formation.",
    "Bevacizumab inhibits VEGF in colorectal cancer. Combination with FOLFOX regimen improves response rates.",
    "Immunocheckpoint inhibitors like Pembrolizumab revolutionize melanoma treatment. PD-L1 expression predicts response.",
    "CAR-T cell therapy for B-cell lymphomas achieves complete remission in 40-50% of refractory cases.",
    "Gene silencing with siRNA targets Huntington's disease. Intrathecal delivery reduces mutant protein levels.",
    "Nanomedicine delivers doxorubicin to breast cancer tumors. Reduced cardiotoxicity compared to conventional chemotherapy.",
    "Gut-brain axis in Parkinson's disease involves alpha-synuclein propagation. Fecal transplants explored as therapy.",
    "Exome sequencing identifies rare variants in autism spectrum disorders. Functional studies validate pathogenicity.",
    "CRISPR screens discover drug resistance genes in cancer. Combination therapies designed to overcome resistance.",
    "Biosimilars for infliximab treat rheumatoid arthritis cost-effectively. Efficacy comparable to originator.",
    "Digital therapeutics for insomnia use CBT-I apps. Improved sleep efficiency without medication.",
    "Pharmacogenomics guides warfarin dosing. CYP2C9 and VKORC1 genotypes predict requirements.",
    "MicroRNA biomarkers detect early pancreatic cancer. Serum panels improve sensitivity over CA19-9 alone.",
    "Regenerative medicine with iPS cells treats macular degeneration. Retinal pigment epithelium transplants restore vision.",
    "Antibody-drug conjugates like Trastuzumab emtansine target HER2-positive breast cancer. Improved progression-free survival.",
    "Viral oncolytics like T-VEC treat melanoma. Intralesional injection induces systemic immunity.",
    
    

    "Ivacaftor significantly improved lung function in cystic fibrosis patients with G551D mutation.",
    "Clinical trials of Ivacaftor demonstrated sustained efficacy over 48 weeks of treatment.",
    "Combination therapy with Ivacaftor and Lumacaftor showed synergistic effects in CFTR modulation.",
    "Patients receiving Ivacaftor reported improved quality of life and reduced pulmonary exacerbations.",
    

    "Low-dose Aspirin is recommended for secondary prevention of cardiovascular events.",
    "Aspirin resistance was observed in approximately 25% of diabetic patients in the study.",
    "The antiplatelet effect of Aspirin reduces thrombus formation in coronary arteries.",
    "Combining Aspirin with Clopidogrel provides dual antiplatelet therapy post-stenting.",
    
    # acetaminophen / paracetamol
    "Acetaminophen remains first-line therapy for mild to moderate pain management.",
    "Paracetamol overdose can cause severe hepatotoxicity requiring N-acetylcysteine treatment.",
    "Acetaminophen combined with NSAIDs provides multimodal analgesia post-surgery.",
    "Paracetamol is preferred over NSAIDs in patients with renal impairment.",
    
    # amoxicillin / penicillin
    "Amoxicillin is first-line treatment for community-acquired pneumonia in outpatients.",
    "Penicillin allergy testing revealed that 90% of reported allergies were not true allergies.",
    "High-dose Amoxicillin overcomes intermediate penicillin resistance in Streptococcus pneumoniae.",
    "Amoxicillin-clavulanate covers beta-lactamase producing respiratory pathogens.",
    
    # ciprofloxacin / levofloxacin
    "Ciprofloxacin is effective against Pseudomonas aeruginosa urinary tract infections.",
    "Levofloxacin penetrates lung tissue well, making it suitable for pneumonia treatment.",
    "Ciprofloxacin should be avoided in patients taking theophylline due to drug interactions.",
    "Levofloxacin-resistant tuberculosis requires alternative multidrug regimens.",
    
    # azithromycin / doxycycline
    "Azithromycin provides convenient once-daily dosing for respiratory infections.",
    "Doxycycline is effective for Lyme disease and rickettsial infections.",
    "Azithromycin has immunomodulatory effects beneficial in chronic lung diseases.",
    "Doxycycline combined with quinine treats chloroquine-resistant malaria.",
    
    # prednisone / dexamethasone
    "Prednisone tapers are used to minimize adrenal suppression after long-term use.",
    "Dexamethasone reduced mortality in severe COVID-19 patients requiring oxygen.",
    "High-dose Prednisone induces remission in autoimmune hepatitis.",
    "Dexamethasone premedication prevents chemotherapy-induced nausea and hypersensitivity.",
    
    # methotrexate
    "Methotrexate is the anchor drug in rheumatoid arthritis treatment algorithms.",
    "Weekly Methotrexate with folic acid supplementation reduces toxicity.",
    "Methotrexate-induced pneumonitis requires immediate drug discontinuation.",
    "High-dose Methotrexate with leucovorin rescue treats osteosarcoma.",
    
    # adalimumab / infliximab / rituximab
    "Adalimumab biosimilars have increased access to TNF-alpha inhibitor therapy.",
    "Infliximab induction therapy achieves rapid remission in ulcerative colitis.",
    "Rituximab depletes CD20-positive B cells in refractory rheumatoid arthritis.",
    "Adalimumab is effective for both Crohn's disease and psoriatic arthritis.",
    "Infliximab requires tuberculosis screening before initiation.",
    
    # trastuzumab / bevacizumab / pembrolizumab / nivolumab
    "Trastuzumab deruxtecan shows activity in HER2-low breast cancer.",
    "Bevacizumab combined with chemotherapy improves survival in ovarian cancer.",
    "Pembrolizumab monotherapy is first-line for PD-L1 high non-small cell lung cancer.",
    "Nivolumab plus ipilimumab combination treats advanced melanoma.",
    "Trastuzumab cardiotoxicity requires regular echocardiographic monitoring.",
    
    # imatinib / erlotinib / sorafenib / lenvatinib
    "Imatinib transformed chronic myeloid leukemia from fatal to manageable disease.",
    "Erlotinib targets EGFR mutations in non-small cell lung cancer.",
    "Sorafenib extends survival in hepatocellular carcinoma patients.",
    "Lenvatinib combined with pembrolizumab treats advanced endometrial cancer.",
    "Imatinib resistance often involves BCR-ABL kinase domain mutations.",
    
    # temozolomide / cisplatin / carboplatin
    "Temozolomide concurrent with radiation is standard for glioblastoma.",
    "Cisplatin-based chemotherapy cures testicular cancer in most patients.",
    "Carboplatin is preferred over cisplatin in elderly patients due to better tolerability.",
    "Temozolomide maintenance continues for six cycles after chemoradiation.",
    
    # gemcitabine / paclitaxel / docetaxel
    "Gemcitabine plus nab-paclitaxel is first-line for metastatic pancreatic cancer.",
    "Paclitaxel causes peripheral neuropathy requiring dose modifications.",
    "Docetaxel improves survival in hormone-refractory prostate cancer.",
    "Weekly Paclitaxel has different toxicity profile than every-3-week dosing.",
    
    # metformin
    "Metformin reduces hepatic glucose production and improves insulin sensitivity.",
    "Metformin should be held before contrast procedures to prevent lactic acidosis.",
    "Metformin is associated with vitamin B12 deficiency with long-term use.",
    "Extended-release Metformin improves gastrointestinal tolerability.",
    
    # atorvastatin / simvastatin / rosuvastatin
    "Atorvastatin 80mg provides intensive lipid lowering post-myocardial infarction.",
    "Simvastatin should not exceed 20mg when combined with amiodarone.",
    "Rosuvastatin achieves greater LDL reduction than equivalent atorvastatin doses.",
    "Atorvastatin has fewer drug interactions than simvastatin.",
    
    # warfarin / apixaban / rivaroxaban / clopidogrel
    "Warfarin requires regular INR monitoring to maintain therapeutic range.",
    "Apixaban has lower bleeding risk than warfarin in atrial fibrillation.",
    "Rivaroxaban once-daily dosing improves medication adherence.",
    "Clopidogrel resistance is associated with CYP2C19 poor metabolizer status.",
    "Warfarin reversal with vitamin K takes 24-48 hours for full effect.",
    
    # omeprazole / pantoprazole
    "Omeprazole inhibits gastric acid secretion by blocking the proton pump.",
    "Pantoprazole has fewer CYP2C19 interactions than omeprazole.",
    "Long-term Omeprazole use increases risk of Clostridium difficile infection.",
    "Pantoprazole is available in intravenous formulation for stress ulcer prophylaxis.",
    
    # sertraline / fluoxetine
    "Sertraline is first-line treatment for panic disorder and social anxiety.",
    "Fluoxetine has the longest half-life among SSRIs, reducing discontinuation syndrome.",
    "Sertraline has fewer drug interactions than fluoxetine or paroxetine.",
    "Fluoxetine is FDA-approved for bulimia nervosa treatment.",
    
    
    # gastrointestinal bleeding
    "Upper gastrointestinal bleeding requires emergent endoscopy within 24 hours.",
    "Gastrointestinal bleeding risk increases with dual antiplatelet therapy.",
    "Proton pump inhibitors reduce recurrent gastrointestinal bleeding after ulcer treatment.",
    "Lower gastrointestinal bleeding in elderly patients often originates from diverticulosis.",
    
    # renal impairment / chronic kidney disease
    "Renal impairment requires dose adjustment of renally cleared medications.",
    "Chronic kidney disease progression is slowed by SGLT2 inhibitors.",
    "Renal impairment contraindicates metformin use when eGFR falls below 30.",
    "Chronic kidney disease causes secondary hyperparathyroidism and bone disease.",
    
    # cardiovascular risk
    "Cardiovascular risk assessment guides statin therapy intensity decisions.",
    "Cardiovascular risk is elevated in rheumatoid arthritis independent of traditional factors.",
    "Reducing cardiovascular risk requires addressing multiple modifiable factors.",
    "Cardiovascular risk calculators underestimate risk in South Asian populations.",
    
    # diabetes mellitus / type 2 diabetes
    "Diabetes mellitus management requires individualized glycemic targets.",
    "Type 2 diabetes prevalence is increasing in adolescent populations.",
    "Diabetes mellitus increases risk of infections and impaired wound healing.",
    "Type 2 diabetes remission is possible with significant weight loss.",
    
    # coronary artery disease / atherosclerosis
    "Coronary artery disease remains the leading cause of mortality worldwide.",
    "Atherosclerosis begins in childhood with fatty streak formation.",
    "Coronary artery disease screening with CT angiography detects subclinical disease.",
    "Atherosclerosis regression occurs with intensive lipid-lowering therapy.",
    
    # osteoporosis
    "Osteoporosis screening with DEXA is recommended for women over 65.",
    "Osteoporosis treatment reduces hip fracture risk by approximately 40%.",
    "Secondary osteoporosis causes include glucocorticoid use and hyperthyroidism.",
    "Osteoporosis in men is underdiagnosed and undertreated.",
    
    # rheumatoid arthritis
    "Rheumatoid arthritis treatment aims for remission or low disease activity.",
    "Early rheumatoid arthritis treatment prevents irreversible joint damage.",
    "Rheumatoid arthritis increases cardiovascular mortality significantly.",
    "Seronegative rheumatoid arthritis has similar outcomes to seropositive disease.",
    
    # alzheimers disease / parkinsons disease
    "Alzheimers disease pathology includes amyloid plaques and tau tangles.",
    "Parkinsons disease motor symptoms respond to dopaminergic therapy.",
    "Alzheimers disease biomarkers in CSF aid early diagnosis.",
    "Parkinsons disease non-motor symptoms often precede motor manifestations.",
    
    # multiple sclerosis
    "Multiple sclerosis relapses are treated with high-dose corticosteroids.",
    "Multiple sclerosis disease-modifying therapies reduce relapse frequency.",
    "Progressive multiple sclerosis has fewer treatment options than relapsing forms.",
    "Multiple sclerosis diagnosis requires dissemination in time and space.",
    
    # hepatitis c
    "Hepatitis C cure rates exceed 95% with direct-acting antiviral regimens.",
    "Hepatitis C screening is recommended for all adults born between 1945-1965.",
    "Chronic Hepatitis C leads to cirrhosis and hepatocellular carcinoma.",
    "Hepatitis C treatment duration is typically 8-12 weeks.",
    
    # covid-19 / influenza / pneumonia / sepsis
    "COVID-19 vaccination significantly reduces hospitalization and death.",
    "Influenza vaccination is recommended annually for all adults.",
    "Community-acquired pneumonia severity guides inpatient versus outpatient treatment.",
    "Sepsis requires early antibiotics and aggressive fluid resuscitation.",
    "COVID-19 long-term sequelae affect multiple organ systems.",
    
    # melanoma / prostate cancer / leukemia / lymphoma
    "Melanoma immunotherapy has dramatically improved survival outcomes.",
    "Prostate cancer screening with PSA remains controversial.",
    "Acute leukemia requires immediate chemotherapy initiation.",
    "Lymphoma staging determines treatment intensity and prognosis.",
    "Melanoma BRAF mutations predict response to targeted therapy.",
    
    # depression / anxiety disorder / schizophrenia
    "Depression screening is recommended in primary care settings.",
    "Anxiety disorder treatment combines pharmacotherapy and psychotherapy.",
    "Schizophrenia requires long-term antipsychotic maintenance therapy.",
    "Treatment-resistant depression may respond to ketamine or ECT.",
    
    # asthma / copd
    "Asthma control is assessed using symptom frequency and reliever use.",
    "COPD exacerbations accelerate lung function decline.",
    "Severe asthma may require biologic therapies targeting specific pathways.",
    "COPD management includes smoking cessation and pulmonary rehabilitation.",
    
    # gout / psoriasis / eczema / migraine / epilepsy
    "Gout flares are triggered by dietary purines and alcohol consumption.",
    "Psoriasis is associated with psoriatic arthritis in 30% of patients.",
    "Eczema management includes emollients and topical anti-inflammatory agents.",
    "Migraine prophylaxis is indicated for frequent or disabling attacks.",
    "Epilepsy surgery can cure drug-resistant focal epilepsy."
    
    
    
       
    
    
    
    
]





class TextDataset(Dataset):
    def __init__(self, texts, word_vocab, char_vocab, max_seq_len=64, max_char_len=20):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        
        self.data = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                tokens = nltk.word_tokenize(sent.lower())
                if len(tokens) < 5: continue
                
                word_ids = [word_vocab.get(t, 0) for t in tokens]
                char_seqs = []
                for t in tokens:
                    chars = list(t)[:max_char_len]
                    char_ids = [char_vocab.get(c, 0) for c in chars] + [0] * (max_char_len - len(chars))
                    char_seqs.append(char_ids)
                
                # Store 3 elements: char IDs, word IDs, original length
                self.data.append((char_seqs, word_ids, len(tokens)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Correcting the unlocking logic here
        char_seq_orig, word_seq_orig, orig_len = self.data[idx]
        
        # Deep copy to avoid modifying the original data
        char_seq = [c[:] for c in char_seq_orig]
        word_seq = word_seq_orig[:]
        
        cur_len = len(word_seq)
        pad_len = self.max_seq_len - cur_len
        
        if pad_len > 0:
            char_seq.extend([[0] * self.max_char_len for _ in range(pad_len)])
            word_seq.extend([0] * pad_len)
        else:
            char_seq = char_seq[:self.max_seq_len]
            word_seq = word_seq[:self.max_seq_len]
        
        return (
            torch.tensor(char_seq, dtype=torch.long),
            torch.tensor(word_seq, dtype=torch.long)
        )



class NERDataset(Dataset):
    """
    NER annotated dataset
    Supports rule-based automatic annotation or loading external annotations
    """
    def __init__(self, texts, word_vocab, char_vocab, tag_vocab,
                 entity_lexicon=None, max_seq_len=64, max_char_len=20):
        """
        Args:
            texts: List[str] raw texts
            word_vocab: dict, word -> id
            char_vocab: dict, char -> id
            tag_vocab: dict, tag -> id (e.g., {'O':0, 'B-DRUG':1, ...})
            entity_lexicon: dict, entity lexicon {'DRUG': [...], 'DISEASE': [...]}
            max_seq_len: maximum sequence length
            max_char_len: maximum character length
        """
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.pad_tag_id = tag_vocab.get('O', 0)
        
        # Default lexicon (can be overridden externally)
        self.entity_lexicon = entity_lexicon or self._default_lexicon()
        
        # Build reverse index for faster lookup
        self._build_entity_index()
        
        # Process data
        self.data = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                processed = self._process_sentence(sent)
                if processed is not None:
                    self.data.append(processed)
    
    def _default_lexicon(self):
        """Default entity lexicon (expandable)"""
        return {
        'DRUG': [
            'ivacaftor', 'aspirin', 'nsaids', 'nsaid', 'statins', 'statin',
            'corticosteroids', 'corticosteroid', 'celecoxib', 'ibuprofen',
            'naproxen', 'crizotinib', 'gefitinib', 'osimertinib', 'afatinib',
            'pt2385', 'probiotic', 'probiotics', 'lactobacillus',
            'acetaminophen', 'paracetamol', 'amoxicillin', 'penicillin', 'ciprofloxacin',
            'levofloxacin', 'azithromycin', 'doxycycline', 'prednisone', 'dexamethasone',
            'methotrexate', 'adalimumab', 'infliximab', 'rituximab', 'trastuzumab',
            'bevacizumab', 'pembrolizumab', 'nivolumab', 'imatinib', 'erlotinib',
            'sorafenib', 'lenvatinib', 'temozolomide', 'cisplatin', 'carboplatin',
            'gemcitabine', 'paclitaxel', 'docetaxel', 'metformin', 'atorvastatin',
            'simvastatin', 'rosuvastatin', 'warfarin', 'apixaban', 'rivaroxaban',
            'clopidogrel', 'omeprazole', 'pantoprazole', 'sertraline', 'fluoxetine'
        ],
        'DISEASE': [
            'cystic fibrosis', 'myocardial infarction', 'heart failure',
            'stroke', 'glioblastoma', 'breast cancer', 'pancreatic cancer',
            'colorectal cancer', 'non-small cell lung cancer', 'lung cancer',
            'ulcerative colitis', 'gastrointestinal bleeding', 'renal impairment',
            'cardiovascular risk', 'hypertensive', 'diabetic',
            'diabetes mellitus', 'type 2 diabetes', 'hypertension', 'coronary artery disease',
            'atherosclerosis', 'osteoporosis', 'rheumatoid arthritis', 'alzheimers disease',
            'parkinsons disease', 'multiple sclerosis', 'chronic kidney disease',
            'hepatitis c', 'covid-19', 'influenza', 'pneumonia', 'sepsis',
            'melanoma', 'prostate cancer', 'leukemia', 'lymphoma', 'depression',
            'anxiety disorder', 'schizophrenia', 'asthma', 'copd', 'gout',
            'psoriasis', 'eczema', 'migraine', 'epilepsy'
        ]
    }
    
    def _build_entity_index(self):
        """Build index for the first words of multi-word entities"""
        self.entity_start_words = {}
        for entity_type, entities in self.entity_lexicon.items():
            for entity in entities:
                tokens = entity.lower().split()
                first_word = tokens[0]
                if first_word not in self.entity_start_words:
                    self.entity_start_words[first_word] = []
                self.entity_start_words[first_word].append((tokens, entity_type))
        
        # Sort by length in descending order to prioritize matching longer entities
        for word in self.entity_start_words:
            self.entity_start_words[word].sort(key=lambda x: len(x[0]), reverse=True)
    
    def _match_entity(self, tokens, start_idx):
        """
        Try to match an entity starting from start_idx
        Returns: (entity_type, length) or (None, 0)
        """
        if start_idx >= len(tokens):
            return None, 0
        
        first_word = tokens[start_idx].lower()
        if first_word not in self.entity_start_words:
            return None, 0
        
        for entity_tokens, entity_type in self.entity_start_words[first_word]:
            entity_len = len(entity_tokens)
            if start_idx + entity_len > len(tokens):
                continue
            
            match = True
            for j, et in enumerate(entity_tokens):
                if tokens[start_idx + j].lower() != et:
                    match = False
                    break
            
            if match:
                return entity_type, entity_len
        
        return None, 0
    
    def _process_sentence(self, sentence):
        """Process a sentence to generate (char_ids, word_ids, tag_ids)"""
        tokens = nltk.word_tokenize(sentence.lower())
        if len(tokens) < 3:
            return None
        
        # Automatic annotation
        tags = ['O'] * len(tokens)
        i = 0
        while i < len(tokens):
            entity_type, entity_len = self._match_entity(tokens, i)
            if entity_type:
                tags[i] = f'B-{entity_type}'
                for j in range(1, entity_len):
                    tags[i + j] = f'I-{entity_type}'
                i += entity_len
            else:
                i += 1
        
        # Convert to IDs
        char_seqs = []
        for t in tokens:
            chars = list(t)[:self.max_char_len]
            c_ids = [self.char_vocab.get(c, 0) for c in chars]
            c_ids += [0] * (self.max_char_len - len(c_ids))
            char_seqs.append(c_ids)
        
        word_ids = [self.word_vocab.get(t, 0) for t in tokens]
        tag_ids = [self.tag_vocab.get(t, 0) for t in tags]
        
        return (char_seqs, word_ids, tag_ids, len(tokens))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        char_seq, word_seq, tag_seq, orig_len = self.data[idx]
        
        cur_len = len(word_seq)
        pad_len = self.max_seq_len - cur_len
        
        if pad_len > 0:
            char_seq = char_seq + [[0] * self.max_char_len] * pad_len
            word_seq = word_seq + [0] * pad_len
            tag_seq = tag_seq + [self.pad_tag_id] * pad_len
        else:
            char_seq = char_seq[:self.max_seq_len]
            word_seq = word_seq[:self.max_seq_len]
            tag_seq = tag_seq[:self.max_seq_len]
            orig_len = self.max_seq_len
        
        # Build mask
        mask = [1.0] * min(orig_len, self.max_seq_len) + [0.0] * max(0, self.max_seq_len - orig_len)
        
        return (
            torch.tensor(char_seq, dtype=torch.long),
            torch.tensor(word_seq, dtype=torch.long),
            torch.tensor(tag_seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float)
        )



    


if __name__ == "__main__":
    print("=" * 60)
    print("ELMo BioNER System - Full Training Process")
    print("=" * 60)
    
    # ------------------------------------------
    # A. Build vocabularies
    # ------------------------------------------
    print("\n[1/5] Building vocabularies...")
    all_text = ' '.join(virtual_dataset)
    tokens = nltk.word_tokenize(all_text.lower())
    words = set(tokens)
    chars = set(''.join(tokens))
    
    word_vocab = {'<PAD>': 0}
    word_vocab.update({w: i+1 for i, w in enumerate(words)})
    
    char_vocab = {'<PAD>': 0}
    char_vocab.update({c: i+1 for i, c in enumerate(chars)})
    
    tag_vocab = {
        'O': 0,
        'B-DRUG': 1, 'I-DRUG': 2,
        'B-DISEASE': 3, 'I-DISEASE': 4
    }
    id2tag = {v: k for k, v in tag_vocab.items()}
    
    CONFIG['word_vocab'] = len(word_vocab)
    CONFIG['char_vocab'] = len(char_vocab)
    
    print(f"   Word vocab: {CONFIG['word_vocab']}, Char vocab: {CONFIG['char_vocab']}")
    
    # ------------------------------------------
    # B. ELMo Pretraining
    # ------------------------------------------
    print("\n[2/5] ELMo BiLM pretraining...")
    
    pretrain_dataset = TextDataset(
        virtual_dataset, word_vocab, char_vocab,
        max_seq_len=CONFIG['seq_len'], max_char_len=CONFIG['char_len']
    )
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    elmo_core = ELMo_BiLM_Core(
        CONFIG['char_vocab'], CONFIG['char_dim'], CONFIG['word_dim'],
        CONFIG['hidden_dim'], CONFIG['word_vocab'], CONFIG['num_layers']
    )
    
    optimizer = optim.Adam(elmo_core.parameters(), lr=1e-3)
    elmo_core.train()
    
    for epoch in range(10):
        total_loss = 0
        for char_batch, word_batch in pretrain_loader:
            optimizer.zero_grad()
            layers, preds = elmo_core(char_batch)
            loss = compute_bi_lm_loss(preds, word_batch, ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(elmo_core.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"   Pretrain Epoch {epoch+1} | Loss: {total_loss/len(pretrain_loader):.4f}")
    
    # ------------------------------------------
    # C. Build NER Dataset
    # ------------------------------------------
    print("\n[3/5] Building NER annotated dataset...")
    
    ner_dataset = NERDataset(
        virtual_dataset, word_vocab, char_vocab, tag_vocab,
        max_seq_len=CONFIG['seq_len'], max_char_len=CONFIG['char_len']
    )
    
    # Split into training/validation sets (80/20)
    train_size = int(0.8 * len(ner_dataset))
    val_size = len(ner_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        ner_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"   Training set: {len(train_dataset)} sentences, Validation set: {len(val_dataset)} sentences")
    
    # ------------------------------------------
    # D. NER Fine-Tuning
    # ------------------------------------------
    print("\n[4/5] Fine-tuning NER model (ELMo frozen)...")
    
    ner_model = BioNER_Model(
        elmo_core, CONFIG['hidden_dim'], len(tag_vocab),
        freeze_elmo=True, pad_idx=0
    )
    
    # Train only ScalarMix + NER layers
    trainable_params = [p for p in ner_model.parameters() if p.requires_grad]
    ner_optimizer = optim.Adam(trainable_params, lr=2e-3)
    
    metrics_calculator = NERMetrics(tag_vocab)
    
    num_epochs = 30
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        ner_model.train()
        train_loss = 0
        
        for char_batch, word_batch, tag_batch, mask_batch in train_loader:
            ner_optimizer.zero_grad()
            loss = ner_model(char_batch, tags=tag_batch, mask=mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            ner_optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        ner_model.eval()
        all_metrics = {'total': {'tp': 0, 'fp': 0, 'fn': 0}}
        
        with torch.no_grad():
            for char_batch, word_batch, tag_batch, mask_batch in val_loader:
                predictions = ner_model(char_batch, mask=mask_batch)
                batch_metrics = metrics_calculator.compute_batch_metrics(
                    predictions, tag_batch, mask_batch
                )
                for key in batch_metrics:
                    if key not in all_metrics:
                        all_metrics[key] = {'tp': 0, 'fp': 0, 'fn': 0}
                    for m in ['tp', 'fp', 'fn']:
                        all_metrics[key][m] += batch_metrics[key][m]
        
        results = metrics_calculator.summary(all_metrics)
        f1 = results['total']['f1']
        
        print(f"   Epoch {epoch+1:2d} | Loss: {avg_train_loss:.4f} | "
              f"P: {results['total']['precision']:.3f} | "
              f"R: {results['total']['recall']:.3f} | "
              f"F1: {f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
    
    print(f"\n   Best F1: {best_f1:.3f}")
    
    # ============================================
    # DEBUG: Insert here (after best F1 print)
    # ============================================
    print("\n" + "=" * 60)
    print("DEBUG: CRF Diagnostic Info")
    print("=" * 60)
    
    # Debug 1: Check CRF transition matrix
    print("\n[Debug 1] CRF Transitions")
    print(f"  Shape: {ner_model.crf.transitions.shape}")
    print(f"  Values:\n{ner_model.crf.transitions.detach().numpy()}")
    print(f"  Start transitions: {ner_model.crf.start_transitions.detach().numpy()}")
    print(f"  End transitions: {ner_model.crf.end_transitions.detach().numpy()}")
    
    # Debug 2: Check mask and sequence length
    print("\n[Debug 2] Mask & Sequence Info")
    for char_batch, word_batch, tag_batch, mask_batch in val_loader:
        print(f"  Batch shape - char: {char_batch.shape}, tag: {tag_batch.shape}, mask: {mask_batch.shape}")
        print(f"  First sequence mask (first 20): {mask_batch[0][:20].tolist()}")
        print(f"  First sequence actual length: {int(mask_batch[0].sum())}")
        print(f"  First sequence tags (first 20): {tag_batch[0][:20].tolist()}")
        break
    
    # Debug 3: Score vs normalizer for a single sample
    print("\n[Debug 3] Score vs Normalizer (single batch)")
    ner_model.eval()
    with torch.no_grad():
        for char_batch, word_batch, tag_batch, mask_batch in val_loader:
            emissions = ner_model._get_emissions(char_batch)
            
            numerator = ner_model.crf._compute_score(emissions, tag_batch, mask_batch)
            denominator = ner_model.crf._compute_normalizer(emissions, mask_batch)
            
            print(f"  Emissions shape: {emissions.shape}")
            print(f"  Emissions range: [{emissions.min():.3f}, {emissions.max():.3f}]")
            print(f"  Numerator (per sample): {numerator[:4].tolist()}")
            print(f"  Denominator (per sample): {denominator[:4].tolist()}")
            print(f"  Loss (per sample): {(denominator - numerator)[:4].tolist()}")
            print(f"  Any negative? {((denominator - numerator) < 0).any().item()}")
            break
    
    # Debug 4: Find features of negative loss samples
    print("\n[Debug 4] Negative Loss Sample Analysis")
    with torch.no_grad():
        for char_batch, word_batch, tag_batch, mask_batch in val_loader:
            emissions = ner_model._get_emissions(char_batch)
            numerator = ner_model.crf._compute_score(emissions, tag_batch, mask_batch)
            denominator = ner_model.crf._compute_normalizer(emissions, mask_batch)
            loss_per_sample = denominator - numerator
            
            # Find negative loss samples
            for i in range(len(loss_per_sample)):
                if loss_per_sample[i] < 0:
                    seq_len = int(mask_batch[i].sum().item())
                    print(f"  Sample {i}:")
                    print(f"    Sequence length: {seq_len}")
                    print(f"    Numerator: {numerator[i].item():.4f}")
                    print(f"    Denominator: {denominator[i].item():.4f}")
                    print(f"    Loss: {loss_per_sample[i].item():.4f}")
                    print(f"    Tags (valid part): {tag_batch[i][:seq_len].tolist()}")
                    print(f"    Emissions range: [{emissions[i, :seq_len].min().item():.2f}, {emissions[i, :seq_len].max().item():.2f}]")
            break
        
    # Debug 5: Trace forward process for problematic sample
    print("\n[Debug 5] Forward Process Trace for Problematic Sample")
    with torch.no_grad():
        for char_batch, word_batch, tag_batch, mask_batch in val_loader:
            emissions = ner_model._get_emissions(char_batch)
            
            # Only look at the 2nd sample (index=1)
            sample_idx = 1
            seq_len = int(mask_batch[sample_idx].sum().item())
            
            e = emissions[sample_idx:sample_idx+1]  # (1, 64, 5)
            m = mask_batch[sample_idx:sample_idx+1]  # (1, 64)
            t = tag_batch[sample_idx:sample_idx+1]   # (1, 64)
            
            crf = ner_model.crf
            
            # Manually run forward, logging each step
            score = crf.start_transitions + e[:, 0]  # (1, 5)
            print(f"  Initial score (pos 0): {score[0].tolist()}")
            print(f"  Initial score max: {score[0].max().item():.2f}, logsumexp: {torch.logsumexp(score[0], dim=0).item():.2f}")
            
            for i in range(1, seq_len):
                broadcast_score = score.unsqueeze(2)
                broadcast_emissions = e[:, i].unsqueeze(1)
                next_score = broadcast_score + crf.transitions + broadcast_emissions
                next_score = torch.logsumexp(next_score, dim=1)
                
                if m[:, i].bool().item():
                    score = next_score
                
                if i <= 5 or i >= seq_len - 2:  # Print start and end
                    print(f"  pos {i}: score max={score[0].max().item():.2f}, logsumexp={torch.logsumexp(score[0], dim=0).item():.2f}")
            
            # Final
            final_score = score + crf.end_transitions
            denominator = torch.logsumexp(final_score, dim=1)
            print(f"  Final score (before end_trans): max={score[0].max().item():.2f}")
            print(f"  Final score (after end_trans): max={final_score[0].max().item():.2f}")
            print(f"  Denominator (logsumexp): {denominator.item():.2f}")
            
            # Compare with numerator
            numerator = crf._compute_score(e, t, m)
            print(f"  Numerator: {numerator.item():.2f}")
            print(f"  Difference: {(denominator - numerator).item():.2f}")
            
            break
        
    print("=" * 60)

    # ------------------------------------------
    # E. Inference Display
    # ------------------------------------------
    print("\n[5/5] Inference Display")
    print("=" * 60)
    
    test_sentences = [
        "Patient diagnosed with cystic fibrosis and treated with Ivacaftor.",
        "The study showed that Aspirin reduces risk of myocardial infarction.",
        "Long-term use of NSAIDs was associated with gastrointestinal bleeding.",
        "Osimertinib demonstrated efficacy in non-small cell lung cancer patients."
    ]
    
    ner_model.eval()
    
    def inference(text):
        tokens = nltk.word_tokenize(text.lower())
        
        char_seqs = []
        for t in tokens:
            chars = list(t)[:CONFIG['char_len']]
            c_ids = [char_vocab.get(c, 0) for c in chars]
            c_ids += [0] * (CONFIG['char_len'] - len(c_ids))
            char_seqs.append(c_ids)
        
        orig_len = len(tokens)
        pad_len = CONFIG['seq_len'] - orig_len
        if pad_len > 0:
            char_seqs += [[0] * CONFIG['char_len']] * pad_len
        else:
            char_seqs = char_seqs[:CONFIG['seq_len']]
            orig_len = CONFIG['seq_len']
        
        char_tensor = torch.tensor([char_seqs], dtype=torch.long)
        mask_tensor = torch.tensor(
            [[1.0] * orig_len + [0.0] * max(0, CONFIG['seq_len'] - orig_len)],
            dtype=torch.float
        )
        
        with torch.no_grad():
            pred_tags = ner_model(char_tensor, mask=mask_tensor)[0]
        
        return tokens[:orig_len], pred_tags[:len(tokens)]
    
    for sent in test_sentences:
        tokens, tags = inference(sent)
        
        formatted_parts = []
        for tk, tag_id in zip(tokens, tags):
            tag = id2tag.get(tag_id, 'O')
            if tag != 'O':
                formatted_parts.append(f"[{tk}/{tag}]")
            else:
                formatted_parts.append(tk)
        
        print(f"\nInput:  {sent}")
        print(f"Output: {' '.join(formatted_parts)}")
    
    print("\n" + "=" * 60)
    print("Training complete! The model has learned to identify DRUG and DISEASE entities.")
    print("=" * 60)


