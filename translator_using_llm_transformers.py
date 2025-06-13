import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
import torch.multiprocessing as mp

# ——— Περιβάλλον & επιδόσεις ——————————————————————————————
# Απενεργοποίηση παραλληλισμού tokenizers για αποφυγή deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ενεργοποίηση cuDNN autotuner για καλύτερη απόδοση σε σταθερά μεγέθη
torch.backends.cudnn.benchmark = True
# Χρήση spawn start method για DataLoader + CUDA σε Python ≥3.8
mp.set_start_method('spawn', force=True)

# ——— Υπερύλικες ρυθμίσεις ——————————————————————————————
MAX_LEN = 128                   # max μήκος σε tokens
VOCAB_SIZE = 4000               # BPE
BATCH_SIZE = 254                # Batch size στο GPU (μετα απο πολλα gpu crashes το 254 αφηνει 200mb ελευθερα στην gpu για να λειτουργεί)
ACCUM_STEPS = 2                 # Gradient accumulation (16×2 = 32 effective batch)
NUM_EPOCHS = 300                 
LEARNING_RATE = 1e-4            # Tempo
EARLY_STOPPING_PATIENCE = 5     # early stopping
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "transformer_translation2.pt" 

# Multi30k dataset split σε train/val/test
raw = load_dataset(
    "bentrevett/multi30k",
    split={"train":"train", "validation":"validation", "test":"test"}
)

# Train BPE Tokenizer. train σε text_iter.Return tokenizer με special tokens, padding & truncation.
def train_tokenizer(text_iter, vocab_size=VOCAB_SIZE):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>","<pad>","<bos>","<eos>"]
    )
    tok.train_from_iterator(text_iter, trainer)
    # post-processing BOS/EOS
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B <eos>",
        special_tokens=[
            ("<bos>", tok.token_to_id("<bos>")),
            ("<eos>", tok.token_to_id("<eos>"))
        ]
    )
    # Padding και truncation
    tok.enable_padding(pad_id=tok.token_to_id("<pad>"),
                      pad_token="<pad>",
                      length=MAX_LEN)
    tok.enable_truncation(max_length=MAX_LEN)
    return tok

# train lang
src_tok = train_tokenizer((ex['de'] for ex in raw['train']))
tgt_tok = train_tokenizer((ex['en'] for ex in raw['train']))

# Save id
SRC_VOCAB_SIZE = src_tok.get_vocab_size()
TGT_VOCAB_SIZE = tgt_tok.get_vocab_size()
PAD_SRC = src_tok.token_to_id("<pad>")
PAD_TGT = tgt_tok.token_to_id("<pad>")

# Dataset/loader ENG DE
class TranslationDataset(Dataset):
    def __init__(self, split):
        self.data = raw[split]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Κωδικοποίηση σε token IDs
        src_ids = src_tok.encode(self.data[idx]['de']).ids
        tgt_ids = tgt_tok.encode(self.data[idx]['en']).ids
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch): # dynamic padding ανά batch Return δύο tensors: (seq_len, batch_size).
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, padding_value=PAD_SRC)
    tgts = nn.utils.rnn.pad_sequence(tgts, padding_value=PAD_TGT)
    return srcs, tgts

# DataLoaders με pin_memory και 2 workers
loader_args = dict(
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2
)
train_loader = DataLoader(TranslationDataset('train'), shuffle=True, **loader_args)
val_loader   = DataLoader(TranslationDataset('validation'), shuffle=False, **loader_args)

# ——— Positional Encoding —————————————————————————————
def get_pos_enc(max_len, d_model):
    """
    sinusoidal positional encoding (max_len, 1, d_model).
    """
    pe = torch.zeros(max_len, d_model, device=DEVICE)
    pos = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=DEVICE) *
        (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(1)

# Transformer Model seq2seq μετάφραση.
class Seq2SeqTransformer(nn.Module):
    def __init__(self, enc_layers, dec_layers, emb_size, nhead,
                 src_vocab, tgt_vocab, ffn_dim=512, drop=0.3):
        super().__init__()
        # Βασικό transformer module
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ffn_dim,
            dropout=drop
        )
        # Ενσωματώσεις (embeddings)
        self.src_emb = nn.Embedding(src_vocab, emb_size)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb_size)
        # Positional encoding
        self.pos_enc = get_pos_enc(MAX_LEN, emb_size)
        # Τελικός χάρτης σε μέγεθος λεξιλογίου
        self.out = nn.Linear(emb_size, tgt_vocab)

    def forward(self, src, tgt):
        # Ενσωμάτωση + positional encoding
        src_e = self.src_emb(src) * math.sqrt(self.src_emb.embedding_dim) \
                + self.pos_enc[:src.size(0)]
        tgt_e = self.tgt_emb(tgt) * math.sqrt(self.tgt_emb.embedding_dim) \
                + self.pos_enc[:tgt.size(0)]

        # Autoregressive mask για decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(0)
        ).to(DEVICE)

        # Padding masks (αγνόηση των <pad> tokens)
        src_pad = (src == PAD_SRC).transpose(0,1)
        tgt_pad = (tgt == PAD_TGT).transpose(0,1)

        # Forward pass
        out = self.transformer(
            src_e,
            tgt_e,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad
        )
        return self.out(out)

scaler = torch.amp.GradScaler()  # mixed precision

@torch.no_grad()
def evaluate(model, loader, criterion): #validation loss
    model.eval()
    total = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:-1], tgt[1:]
        # Mixed precision inference
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logits = model(src, tgt_in)
            total += criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1)
            ).item()
    return total / len(loader)

def train_epoch(model, loader, optimizer, criterion): #gradient accumulation
    model.train()
    total = 0
    optimizer.zero_grad()
    for i, (src, tgt) in enumerate(loader, 1):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:-1], tgt[1:]
        # Mixed precision training
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logits = model(src, tgt_in)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1)
            ) / ACCUM_STEPS
        scaler.scale(loss).backward()
        # update ACCUM_STEPS batches
        if i % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)

# Beam Search Decode after-training inference με beam search. Δίνει πιο accurate μεταφράσεις από greedy decode.
@torch.no_grad()
def beam_search_decode(model, sentence, beam_size=5, max_len=MAX_LEN):
    model.eval()
    # Tokenize + μεταφορά σε cuda
    src_ids = src_tok.encode(sentence.lower()).ids
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(1)

    # Encode source για memory vector
    memory = model.transformer.encoder(
        model.src_emb(src) + model.pos_enc[:src.size(0)],
        src_key_padding_mask=(src == PAD_SRC).transpose(0,1)
    )

    # beams BOS token
    sequences = [(torch.tensor([tgt_tok.token_to_id("<bos>")],
                               device=DEVICE), 0.0)]

    for _ in range(max_len):
        all_cand = []
        for seq, score in sequences:
            # If EOS exists goto final stage
            if seq[-1].item() == tgt_tok.token_to_id("<eos>"):
                all_cand.append((seq, score))
                continue
            #  decoder input
            tgt = seq.unsqueeze(1)
            tgt_mask = model.transformer.generate_square_subsequent_mask(
                tgt.size(0)
            ).to(DEVICE)
            # Decoder pass
            out = model.transformer.decoder(
                model.tgt_emb(tgt) + model.pos_enc[:tgt.size(0)],
                memory,
                tgt_mask=tgt_mask
            )
            logits = model.out(out[-1])
            probs = torch.log_softmax(logits, dim=-1)
            # Best beam_size tokens
            topk_vals, topk_idxs = torch.topk(probs, beam_size)

            # Επέκταση κάθε partial sequence
            for i in range(beam_size):
                nid = topk_idxs[0][i].item()
                cand = torch.cat([seq, torch.tensor([nid], device=DEVICE)])
                cand_score = score + topk_vals[0][i].item()
                all_cand.append((cand, cand_score))

        # Save best beam_size sequences
        sequences = sorted(all_cand, key=lambda x: x[1], reverse=True)[:beam_size]

    # Ret best seq
    best_seq = sequences[0][0]
    return tgt_tok.decode(best_seq.tolist())
if __name__ == "__main__":
    # Main
    # model, optimizer, criterion με label smoothing
    model = Seq2SeqTransformer(
        enc_layers=2, dec_layers=2, emb_size=256, nhead=4,
        src_vocab=SRC_VOCAB_SIZE, tgt_vocab=TGT_VOCAB_SIZE
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_TGT,
        label_smoothing=0.1
    )

    # Load saved Model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded trained model from disk.")
    else:
        # training
        best_val = float('inf')
        epochs_no_improve = 0
        for epoch in range(1, NUM_EPOCHS+1):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch}: Train={tr_loss:.3f}, Val={val_loss:.3f}")

            # Save model κ Early stopping
            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print("New best model saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    while True:
        sent = input("Enter German sentence (or 'quit'): ")
        if sent.lower() in {'quit','exit'}:
            break
        print("Translation:", beam_search_decode(model, sent))

