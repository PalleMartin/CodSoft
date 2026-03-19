"""
=============================================================
  IMAGE CAPTIONING — Encoder-Decoder with ResNet + LSTM
  Internship Task Implementation
=============================================================

OVERVIEW:
  - Encoder : ResNet-50 (pretrained on ImageNet) extracts a
              2048-dim feature vector from each input image.
  - Decoder : LSTM takes that vector as its initial hidden
              state, then generates caption tokens one by one.
  - Training: Teacher forcing — feed the ground-truth word
              at each step rather than the predicted word,
              so gradients flow reliably in early epochs.

REQUIREMENTS:
  pip install torch torchvision pillow matplotlib
"""

# ─── Imports ───────────────────────────────────────────────
import os
import re
import json
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── 1. VOCABULARY ──────────────────────────────────────────
class Vocabulary:
    """
    Maps words ↔ integer indices.

    Special tokens:
      <PAD>   – padding to make batches the same length
      <START> – prepended to every caption during decoding
      <END>   – tells the decoder to stop
      <UNK>   – replaces rare / out-of-vocabulary words
    """

    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self, freq_threshold: int = 1):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    # ── build from a list of caption strings ──
    def build(self, captions: list[str]) -> None:
        counts = Counter()
        for cap in captions:
            counts.update(self._tokenize(cap))

        idx = len(self.word2idx)
        for word, freq in counts.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # ── encode a caption string → list of ints ──
    def encode(self, caption: str) -> list[int]:
        return (
            [self.START]
            + [self.word2idx.get(w, self.UNK) for w in self._tokenize(caption)]
            + [self.END]
        )

    # ── decode a list of ints → sentence string ──
    def decode(self, indices: list[int], skip_special: bool = True) -> str:
        special = {self.PAD, self.START, self.END}
        words = [
            self.idx2word.get(i, "<UNK>")
            for i in indices
            if not (skip_special and i in special)
        ]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.word2idx, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


# ─── 2. DATASET ─────────────────────────────────────────────
class CaptionDataset(Dataset):
    """
    Expects:
      image_dir  – folder of images
      captions   – list of (filename, caption_string) tuples
      vocab      – a built Vocabulary instance

    Returns (image_tensor, caption_tensor) pairs.
    """

    # ImageNet normalization — required because ResNet was trained on these stats
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        image_dir: str,
        captions: list[tuple[str, str]],
        vocab: Vocabulary,
        transform=None,
    ):
        self.image_dir = image_dir
        self.captions = captions
        self.vocab = vocab
        self.transform = transform or self.TRANSFORM

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int):
        filename, caption = self.captions[idx]
        img_path = os.path.join(self.image_dir, filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return image, tokens


def collate_fn(batch):
    """
    Pad variable-length captions within a batch to the same length.
    PyTorch's DataLoader calls this to combine samples into a batch tensor.
    """
    images, captions = zip(*batch)
    images = torch.stack(images, 0)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, : len(cap)] = cap

    return images, padded, torch.tensor(lengths, dtype=torch.long)


# ─── 3. ENCODER (CNN) ───────────────────────────────────────
class EncoderCNN(nn.Module):
    """
    ResNet-50 with the final classification head replaced by a
    linear layer that projects the 2048-d pool output to
    embed_size dimensions — the same size the decoder expects.

    We freeze all conv layers at first; only train the new
    linear projection. Later you can unfreeze for fine-tuning.
    """

    def __init__(self, embed_size: int = 256, fine_tune: bool = False):
        super().__init__()

        # Load ImageNet-pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Drop the final fully-connected classification layer
        modules = list(resnet.children())[:-1]   # keep everything except fc
        self.resnet = nn.Sequential(*modules)

        # New projection: 2048 → embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Optionally allow gradients to flow through ResNet
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images : (batch, 3, 224, 224)
        returns: (batch, embed_size)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.resnet(images)       # (batch, 2048, 1, 1)

        features = features.reshape(features.size(0), -1)   # (batch, 2048)
        features = self.bn(self.fc(features))                # (batch, embed_size)
        return features


# ─── 4. DECODER (RNN) ───────────────────────────────────────
class DecoderRNN(nn.Module):
    """
    Single-layer LSTM language model conditioned on image features.

    Architecture:
      embed       – word embeddings (vocab_size → embed_size)
      lstm        – unrolled over the caption tokens
      linear      – projects hidden state → vocab logits
      dropout     – regularisation

    At each step t the LSTM receives:
      input  = embedding of word at t-1
      hidden = (h_{t-1}, c_{t-1})

    The very first hidden state h_0 is initialised from the image
    feature vector via a learnable linear layer (init_hidden).
    """

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embed   = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm    = nn.LSTM(embed_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear  = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Project image features → (h_0, c_0)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    # ── called during training (teacher forcing) ──
    def forward(
        self, features: torch.Tensor, captions: torch.Tensor
    ) -> torch.Tensor:
        """
        features  : (batch, embed_size)  ← from EncoderCNN
        captions  : (batch, seq_len)     ← integer token ids, includes <START>
        returns   : (batch, seq_len, vocab_size) logits
        """
        # Initialise LSTM hidden state from image features
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)  # (1, batch, hidden)
        c0 = torch.tanh(self.init_c(features)).unsqueeze(0)

        # Embed caption tokens (skip the last <END> token — we predict it)
        embeddings = self.dropout(self.embed(captions[:, :-1]))  # (batch, seq-1, embed)

        # Unroll LSTM over the embedded sequence
        hiddens, _ = self.lstm(embeddings, (h0, c0))   # (batch, seq-1, hidden)

        # Project to vocabulary logits
        outputs = self.linear(hiddens)                  # (batch, seq-1, vocab)
        return outputs

    # ── called at inference (greedy / beam search) ──
    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        vocab: Vocabulary,
        max_len: int = 30,
        device: str = "cpu",
    ) -> str:
        """
        Autoregressively generates a caption for one image feature vector.
        Uses greedy decoding (picks argmax at each step).
        """
        h = torch.tanh(self.init_h(features)).unsqueeze(0)
        c = torch.tanh(self.init_c(features)).unsqueeze(0)

        word_id = torch.tensor([[vocab.START]], device=device)
        result  = []

        for _ in range(max_len):
            emb        = self.embed(word_id)               # (1, 1, embed)
            output, (h, c) = self.lstm(emb, (h, c))
            logits     = self.linear(output.squeeze(1))    # (1, vocab)
            word_id    = logits.argmax(1, keepdim=True)    # greedy pick

            token = word_id.item()
            if token == vocab.END:
                break
            result.append(token)

        return vocab.decode(result)


# ─── 5. FULL MODEL ──────────────────────────────────────────
class ImageCaptioningModel(nn.Module):
    """Convenience wrapper that chains Encoder → Decoder."""

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs  = self.decoder(features, captions)
        return outputs

    def caption(self, image_tensor, vocab, max_len=30, device="cpu"):
        """End-to-end inference: image tensor → caption string."""
        self.eval()
        features = self.encoder(image_tensor.unsqueeze(0).to(device))
        return self.decoder.generate(features, vocab, max_len, device)


# ─── 6. TRAINING LOOP ───────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, vocab):
    model.train()
    total_loss = 0

    for batch_idx, (images, captions, lengths) in enumerate(loader):
        images   = images.to(device)
        captions = captions.to(device)

        # Forward pass
        outputs = model(images, captions)          # (batch, seq-1, vocab)

        # Flatten for cross-entropy
        # target = captions shifted left by one (we predict next token)
        targets = captions[:, 1:]                  # (batch, seq-1)

        # Mask padding so it doesn't contribute to loss
        mask = targets != vocab.PAD                # boolean (batch, seq-1)
        loss = criterion(
            outputs[mask],         # (N_real_tokens, vocab)
            targets[mask],         # (N_real_tokens,)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, vocab):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions, lengths in loader:
            images   = images.to(device)
            captions = captions.to(device)
            outputs  = model(images, captions)
            targets  = captions[:, 1:]
            mask     = targets != vocab.PAD
            loss     = criterion(outputs[mask], targets[mask])
            total_loss += loss.item()
    return total_loss / len(loader)


def train(
    model,
    train_loader,
    val_loader,
    vocab,
    num_epochs: int = 10,
    lr: float = 3e-4,
    device: str = "cpu",
    checkpoint_path: str = "caption_model.pth",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device, vocab)
        val_loss = evaluate(model, val_loader, criterion, device, vocab)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")

    return train_losses, val_losses


# ─── 7. INFERENCE HELPER ────────────────────────────────────
def caption_image(
    image_path: str,
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    device: str = "cpu",
) -> str:
    """Load an image from disk and return its generated caption."""
    transform = CaptionDataset.TRANSFORM
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return model.caption(image_tensor, vocab, device=device)


def visualize_prediction(image_path, caption, save_path=None):
    """Display the image with its predicted caption."""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Caption: {caption}", fontsize=14, wrap=True, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─── 8. EVALUATION METRIC — BLEU ────────────────────────────
def bleu_score(reference: list[str], hypothesis: list[str], n: int = 4) -> float:
    """
    Compute sentence-level BLEU-n score.
    BLEU measures n-gram overlap between generated and reference captions.
    Score of 1.0 = perfect match, 0.0 = no overlap.
    """
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def clipped_precision(ref, hyp, n):
        ref_counts = Counter(ngrams(ref, n))
        hyp_counts = Counter(ngrams(hyp, n))
        clipped   = sum(min(c, ref_counts[ng]) for ng, c in hyp_counts.items())
        total     = max(len(ngrams(hyp, n)), 1)
        return clipped / total

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(reference) / max(len(hypothesis), 1)))

    log_sum = 0.0
    for i in range(1, n + 1):
        p = clipped_precision(reference, hypothesis, i)
        log_sum += math.log(p + 1e-10)

    return bp * math.exp(log_sum / n)


# ─── 9. DEMO / QUICK TEST ───────────────────────────────────
def quick_demo():
    """
    Builds a tiny toy vocabulary and model, runs a single forward
    pass, then generates a dummy caption — no real data needed.
    """
    print("=" * 55)
    print("  IMAGE CAPTIONING — Quick Architecture Demo")
    print("=" * 55)

    # Dummy captions for vocabulary building
    captions_raw = [
        "a dog running on the grass",
        "a cat sitting on a chair",
        "two people walking in the park",
        "a red car parked near the road",
        "children playing with a ball",
    ]

    vocab = Vocabulary(freq_threshold=1)
    vocab.build(captions_raw)
    print(f"\nVocabulary size: {len(vocab)}")

    # Model config
    EMBED_SIZE  = 128
    HIDDEN_SIZE = 256
    VOCAB_SIZE  = len(vocab)
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")

    model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
    model.to(DEVICE)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params   : {total:,}")
    print(f"Trainable params: {train:,}")

    # Forward pass with random batch
    batch_size = 2
    seq_len    = 8
    dummy_imgs  = torch.randn(batch_size, 3, 224, 224).to(DEVICE)
    dummy_caps  = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)).to(DEVICE)
    dummy_caps[:, 0] = vocab.START

    outputs = model(dummy_imgs, dummy_caps)
    print(f"\nForward pass output shape: {outputs.shape}")
    print(f"Expected: (batch={batch_size}, seq-1={seq_len-1}, vocab={VOCAB_SIZE})")

    # Generate caption (greedy)
    sample_img = torch.randn(3, 224, 224)
    generated  = model.caption(sample_img, vocab, device=DEVICE)
    print(f"\nSample generated caption (random weights): '{generated}'")
    print("\n✓ Architecture verified — train with real data for meaningful captions.")

    # BLEU demo
    ref = "a dog running on the grass".split()
    hyp = "a dog playing on the lawn".split()
    score = bleu_score(ref, hyp, n=4)
    print(f"\nBLEU-4 demo — ref: '{' '.join(ref)}'")
    print(f"             hyp: '{' '.join(hyp)}'")
    print(f"             score: {score:.4f}")

    print("\n" + "=" * 55)
    print("  HOW TO TRAIN ON REAL DATA")
    print("=" * 55)
    print("""
1. Download MS-COCO 2017 dataset
   https://cocodataset.org/#download
   (train2017 images + captions/annotations)

2. Parse annotations:
   with open('captions_train2017.json') as f:
       data = json.load(f)
   captions = [(a['image_id'], a['caption']) for a in data['annotations']]

3. Build vocabulary:
   vocab = Vocabulary(freq_threshold=5)
   vocab.build([c for _, c in captions])

4. Create dataset + dataloader:
   dataset = CaptionDataset('train2017/', captions, vocab)
   loader  = DataLoader(dataset, batch_size=64,
                        collate_fn=collate_fn, shuffle=True)

5. Train:
   model = ImageCaptioningModel(256, 512, len(vocab), num_layers=2)
   train(model, train_loader, val_loader, vocab,
         num_epochs=30, lr=3e-4, device='cuda')

6. Caption a new image:
   result = caption_image('my_photo.jpg', model, vocab, device='cuda')
   print(result)
""")


if __name__ == "__main__":
    quick_demo()