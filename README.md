# Encoder-Decoder Translation System

A Transformer encoder-decoder model built in PyTorch that translates English into German. The notebook covers the whole pipeline: loading and preprocessing the data, building the vocabularies, building the model, training it, and translating sentences.

## Results

Trained on 135,000 English-German sentence pairs (validated on 15,000) for 2 epochs:

| Epoch | Train loss | Validation loss |
|-------|-----------|-----------------|
| 1     | 2.412     | 1.578           |
| 2     | 1.471     | 1.313           |

Sample translations:

| English    | Reference       | Model output       |
|------------|-----------------|--------------------|
| go .       | geh .           | geh !              |
| go away .  | gehen sie weg . | geh weg .          |
| show me .  | zeig's mir !    | zeigen sie mir !   |

Training for more epochs gives better translations.

## Model

| Setting | Value |
|---------|-------|
| Embedding size (`d_model`) | 256 |
| Attention heads | 8 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Feed-forward size | 512 |
| Dropout | 0.1 |
| Max sequence length | 40 |
| Parameters | about 12M |

- **Encoder:** token embedding, sinusoidal positional encoding, dropout, and `nn.TransformerEncoder` layers. Padding tokens are masked out.
- **Decoder:** scaled token embedding, positional encoding, masked self-attention (a causal mask hides future tokens), cross-attention to the encoder output, and a linear layer that projects onto the German vocabulary.
- **Training:** Adam (lr = 0.001) with cross-entropy loss, ignoring padding tokens.
- **Inference:** greedy autoregressive decoding starting from `<sos>` and stopping at `<eos>`, with optional temperature sampling.

## Project structure

```
Encoder-Decoder/
├── C3M3_Assignment (1).ipynb   # Main notebook: data prep, model, training, translation
├── helper_utils.py             # Dataset loading, normalization/tokenization, training loop, plots
├── unittests.py                # Tests for the Encoder, Decoder and EncoderDecoder classes
└── languages/                  # Dataset files (not included, see below)
```

## Setup

1. Install the dependencies:

   ```bash
   pip install torch numpy matplotlib tqdm jupyter
   ```

   `unittests.py` also needs the `dlai_grader` package. That package comes from the DeepLearning.AI course environment. Without it, skip the `unittests.exercise_*` cells.

2. Get the dataset. Create a `languages/` folder next to the notebook and put a tab-separated English-to-target file in it, such as `deu-eng.zip` or `deu.txt` for German. These files use the Tatoeba / [ManyThings.org Anki](https://www.manythings.org/anki/) format. Zip files are extracted automatically.

   Supported languages: French (`fra`), Spanish (`spa`), German (`deu`), Italian (`ita`), Portuguese (`por`), Russian (`rus`).

3. Run the notebook:

   ```bash
   jupyter notebook "C3M3_Assignment (1).ipynb"
   ```

   When asked, enter the number of the target language (`3` for German). A GPU is used automatically if one is available.

## Notebook sections

1. Setup
2. Loading and preprocessing the data
3. Building the vocabulary
4. Converting sentences to indices
5. Dataset and DataLoaders
6. Masks (padding and causal)
7. Positional encoding
8. Encoder
9. Decoder
10. Encoder-decoder model
11. Building the full model
12. Training
13. Translation
