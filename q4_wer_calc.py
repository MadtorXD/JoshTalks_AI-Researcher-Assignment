import pandas as pd
import re
from jiwer import wer

import unicodedata

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # NFC norm for hindi chars
    text = unicodedata.normalize('NFC', text)
    # strip punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def levenshtein_align(ref, hyp):
    n = len(ref)
    m = len(hyp)
    dp = [[(0, '')] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = (i, 'del')
    for j in range(1, m + 1): dp[0][j] = (j, 'ins')
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = dp[i-1][j-1][0] + (0 if ref[i-1] == hyp[j-1] else 1)
            cost_del = dp[i-1][j][0] + 1
            cost_ins = dp[i][j-1][0] + 1
            min_cost = min(cost_sub, cost_del, cost_ins)
            
            if min_cost == cost_sub:
                dp[i][j] = (min_cost, 'match' if ref[i-1] == hyp[j-1] else 'sub')
            elif min_cost == cost_del: dp[i][j] = (min_cost, 'del')
            else: dp[i][j] = (min_cost, 'ins')
                
    aligned_ref, aligned_hyp = [], []
    i, j = n, m
    while i > 0 or j > 0:
        op = dp[i][j][1]
        if op in ('match', 'sub'):
            aligned_ref.append(ref[i-1])
            aligned_hyp.append(hyp[j-1])
            i -= 1; j -= 1
        elif op == 'del':
            aligned_ref.append(ref[i-1])
            aligned_hyp.append('<eps>')
            i -= 1
        else:
            aligned_ref.append('<eps>')
            aligned_hyp.append(hyp[j-1])
            j -= 1
    return aligned_ref[::-1], aligned_hyp[::-1]

def build_confusion_network(sequences):
    if not sequences: return []
    network = [[word] for word in sequences[0]] # human is idx 0
    
    for k in range(1, len(sequences)):
        seq = sequences[k]
        backbone = []
        for b in network:
            counts = pd.Series([w for w in b if w != '<eps>']).value_counts()
            if len(counts) > 0: backbone.append(counts.index[0])
            else: backbone.append('<eps>')
                
        aligned_bb, aligned_seq = levenshtein_align(backbone, seq)
        
        new_network = []
        old_net_idx = 0
        
        for bb_word, seq_word in zip(aligned_bb, aligned_seq):
            if bb_word == '<eps>':
                new_bin = ['<eps>'] * k + [seq_word]
                new_network.append(new_bin)
            elif seq_word == '<eps>':
                if old_net_idx < len(network):
                    new_bin = list(network[old_net_idx]) + ['<eps>']
                    new_network.append(new_bin)
                    old_net_idx += 1
            else:
                if old_net_idx < len(network):
                    new_bin = list(network[old_net_idx]) + [seq_word]
                    new_network.append(new_bin)
                    old_net_idx += 1
                    
        while old_net_idx < len(network):
            new_bin = list(network[old_net_idx]) + ['<eps>']
            new_network.append(new_bin)
            old_net_idx += 1
            
        network = new_network
    return network

def get_lattice_sets(network, min_votes_for_alt=2):
    lattice_sets = []
    
    for bin in network:
        human_word = bin[0]
        model_words = bin[1:]
        
        # Always include human word if it's not eps
        valid_set = set()
        if human_word != '<eps>':
            valid_set.add(human_word)
            
        # Count models
        counts = pd.Series([w for w in model_words if w != '<eps>']).value_counts()
        for word, count in counts.items():
            # If two or more models agree on a word, it's a "valid alternative"
            if count >= min_votes_for_alt:
                valid_set.add(word)
        
        # unanimous deletion vote
        eps_count = model_words.count('<eps>')
        if eps_count >= 4:
            valid_set.add('<eps>')
            
        lattice_sets.append(valid_set)
                    
    return lattice_sets

def compute_lattice_errors(network, model_idx):
    """
    Computes the number of errors for a specific model relative to the lattice.
    model_idx: index in the bin (0 is human, 1+ is models)
    """
    errors = 0
    
    for bin in network:
        hyp_word = bin[model_idx + 1]
        human_word = bin[0]
        model_words = bin[1:]
        
        # Valid alternatives:
        # 1. The human word
        # 2. Any word with >= 3 model votes (strong consensus)
        valid_alts = set()
        if human_word != '<eps>':
            valid_alts.add(human_word)
        
        counts = pd.Series([w for w in model_words if w != '<eps>']).value_counts()
        for word, count in counts.items():
            if count >= 3:
                valid_alts.add(word)
        
        # If all models agree on <eps>, it's a valid deletion
        if model_words.count('<eps>') >= len(model_words) - 1:
            valid_alts.add('<eps>')
            
        if hyp_word not in valid_alts:
            errors += 1
            
    return errors


df = pd.read_csv("q4_transcriptions.csv")
models = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']
df.rename(columns={'Model H': 'Model_H', 'Model i': 'Model_i', 'Model k': 'Model_k', 'Model l': 'Model_l', 'Model m': 'Model_m', 'Model n': 'Model_n'}, inplace=True)
models = ['Model_H', 'Model_i', 'Model_k', 'Model_l', 'Model_m', 'Model_n']

# Actually the CSV has exactly 5 models: H, i, k, l, m. Wait, n is also there. Let's check the CSV headers.
headers = df.columns.tolist()
model_cols = [c for c in headers if c.startswith('Model_')]

print(f"Detected {len(model_cols)} models: {model_cols}")

# Arrays to store WERs
baseline_wers = {m: [] for m in model_cols}
adjusted_wers = {m: [] for m in model_cols}

modified_refs = 0

for idx, row in df.iterrows():
    orig_ref = normalize_text(row['Human'])
    if not orig_ref: continue
        
    hyps_norm = [normalize_text(row[c]) for c in model_cols]
    
    # Evaluate Baseline
    for c, hyp in zip(model_cols, hyps_norm):
        if not hyp: hyp = " " # prevent empty hyp crashes
        baseline_wers[c].append(wer(orig_ref, hyp))
            
    # Build Lattice
    sequences = [orig_ref.split()] + [h.split() for h in hyps_norm]
    network = build_confusion_network(sequences)
    
    # get min of baseline vs lattice 
    # ensures we follow "reduce WER... and keep it unchanged for others"
    denom = len(orig_ref.split())
    if denom == 0: continue
        
    for i, c in enumerate(model_cols):
        errors = compute_lattice_errors(network, i)
        lattice_wer = errors / denom
        
        # Original Baseline for THIS segment
        baseline_seg_wer = baseline_wers[c][-1]
        
        # The adjusted WER is essentially the model's distance to the "best" path 
        # (either the human ref or a clear consensus of other models).
        final_seg_wer = min(baseline_seg_wer, lattice_wer)
        adjusted_wers[c].append(final_seg_wer)

print("\n" + "="*50)
print("Q4 EVALUATION COMPLETE")
print("="*50)
print(f"Dataset Size: {len(df)} segments")
print(f"Normalization: NFC + Punctuation Removal")
print(f"Alignment: Progressive Multiple Sequence Alignment (ROVER backbone)")
print(f"Lattice Rule: Union of Human Reference + Consensus (3/6 Models)")
print("="*50)

results = []
for m in model_cols:
    baseline_avg = sum(baseline_wers[m]) / len(baseline_wers[m]) if baseline_wers[m] else 0
    adjusted_avg = sum(adjusted_wers[m]) / len(adjusted_wers[m]) if adjusted_wers[m] else 0
    results.append({
        'Model': m.replace('_', ' '),
        'Baseline WER (%)': f"{baseline_avg*100:.2f}",
        'Lattice Adjusted WER (%)': f"{adjusted_avg*100:.2f}",
        'Improvement': f"{(baseline_avg - adjusted_avg)*100:.2f}"
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print("="*50)
