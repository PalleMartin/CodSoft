Image Captioning using ResNet + LSTM
Overview

This project implements an Image Captioning system using a deep learning encoder–decoder architecture. The model takes an image as input and generates a natural language description of its content.

The architecture combines:

CNN (ResNet-50) → Extracts visual features from images

RNN (LSTM) → Generates captions word-by-word

This is a standard, industry-relevant approach used in real-world vision-language systems.

Architecture
1. Encoder (CNN)

Model: Pretrained ResNet-50

Removes final classification layer

Outputs a 2048-dimensional feature vector

Projects features to embedding size

2. Decoder (LSTM)

Takes image features as initial hidden state

Uses embeddings + LSTM to generate sequence

Outputs probability distribution over vocabulary

3. Training Strategy

Teacher Forcing is used

Ground-truth tokens are fed during training

Cross-entropy loss with padding masking

Project Structure
├── Vocabulary          # Word ↔ Index mapping
├── Dataset            # Image + Caption loader
├── EncoderCNN         # ResNet-based feature extractor
├── DecoderRNN         # LSTM caption generator
├── ImageCaptioningModel  # Full pipeline
├── Training Loop      # Training + validation
├── BLEU Score         # Evaluation metric
├── Demo               # Quick test run
Installation

Install dependencies:

pip install torch torchvision pillow matplotlib
Dataset

Recommended dataset:

MS COCO 2017

Download:
https://cocodataset.org/#download

You need:

train2017/ (images)

captions_train2017.json (annotations)

How It Works
Step 1: Build Vocabulary
vocab = Vocabulary(freq_threshold=5)
vocab.build(captions_list)
Step 2: Create Dataset
dataset = CaptionDataset(image_dir, captions, vocab)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
Step 3: Initialize Model
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab))
Step 4: Train
train(model, train_loader, val_loader, vocab, num_epochs=30, device='cuda')
Step 5: Inference
caption = caption_image("test.jpg", model, vocab, device='cuda')
print(caption)
Key Concepts (Don’t Ignore These)
1. Teacher Forcing

Helps stabilize training early

Without it → model fails to learn sequence structure

Example 1

Input: <START> a dog running

Model predicts next words step-by-step using ground truth

Example 2

Without teacher forcing → model feeds its own wrong predictions → error compounds

2. Padding & Masking

Captions have variable length

Padding ensures batch processing

Example 1

[a dog running]
[a cat]
→ padded → [a cat <PAD> <PAD>]

Example 2

Loss ignores <PAD> tokens using masking

3. Feature Conditioning

Image vector initializes LSTM hidden state

Example 1

Image of dog → hidden state biased toward "dog-related words"

Example 2

Without conditioning → model generates random generic captions

Evaluation
BLEU Score

Measures overlap between generated and reference captions.

Range: 0 → 1

Higher = better

Example 1

Reference: "a dog running on grass"

Prediction: "a dog running on grass" → BLEU ≈ 1.0

Example 2

Prediction: "a cat sitting inside house" → BLEU ≈ low

Quick Demo

Run:

python your_script.py

What it does:

Builds toy vocabulary

Runs forward pass

Generates dummy caption

Computes BLEU score

Limitations (Be Real About This)

Greedy Decoding is weak

Produces repetitive or suboptimal captions

Use Beam Search for improvement

No Attention Mechanism

Model sees entire image as one vector

Misses spatial details

Overfitting Risk

Small dataset → poor generalization

Improvements (If You Want This to Stand Out)
1. Add Attention

Improves caption quality significantly

Example:

Focus on “dog” region instead of entire image

Focus shifts word-by-word

2. Use Beam Search

Generates better sequences than greedy decoding

Example:

Greedy: “a dog is is running”

Beam: “a dog is running in the park”

Conclusion

This project demonstrates:

CNN + RNN integration

Sequence modeling

Real-world AI pipeline

If you understand this end-to-end, you're already above most beginners.

If you just copied it without understanding:
→ you’ll fail in interviews instantly.
