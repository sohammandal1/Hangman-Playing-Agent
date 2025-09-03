# File: canine_model.py
import torch
import torch.nn as nn
from transformers import CanineModel, CanineConfig

class HangmanCANINE(nn.Module):
    def __init__(self, max_len=20, guess_vector_dim=26, hidden_dim=128):
        super(HangmanCANINE, self).__init__()
        config = CanineConfig.from_pretrained("google/canine-c")
        config.num_labels = 26  # 26 letters

        self.canine = CanineModel(config)
        self.max_len = max_len

        # Additional layers
        self.guess_encoder = nn.Linear(guess_vector_dim, hidden_dim)
        self.combined_proj = nn.Linear(config.hidden_size + hidden_dim, 256)
        self.classifier = nn.Linear(256, 26)

    def forward(self, input_ids, guess_vector):
        canine_out = self.canine(input_ids=input_ids).last_hidden_state
        pooled = canine_out[:, 0, :]  # CLS token embedding

        guess_encoded = self.guess_encoder(guess_vector)
        combined = torch.cat([pooled, guess_encoded], dim=1)

        logits = self.classifier(torch.relu(self.combined_proj(combined)))
        return logits


# Example instantiation
if __name__ == '__main__':
    model = HangmanCANINE()
    dummy_input = torch.randint(0, 256, (2, 20))  # assuming byte-level ids
    dummy_guesses = torch.rand(2, 26)
    out = model(dummy_input, dummy_guesses)
    print("Model output shape:", out.shape)
