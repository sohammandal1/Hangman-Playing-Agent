# Hangman AI with CANINE Transformer and PPO

This project implements an AI agent that learns to play Hangman using Google's CANINE transformer model combined with Proximal Policy Optimization (PPO) reinforcement learning. The agent processes character-level representations of partially revealed words and previous guesses to predict the next optimal letter.

## Features

- **CANINE Transformer Integration**: Uses Google's character-level CANINE model for robust text understanding
- **Reinforcement Learning**: PPO agent learns optimal letter selection strategies
- **Custom Environment**: Complete Hangman game environment with configurable rules
- **Comprehensive Evaluation**: Tools for training, inference, and model evaluation
- **Flexible Data Processing**: Handles word encoding, guess tracking, and state representation

## Project Structure

```
├── canine_model.py          # CANINE-based neural network architecture
├── data_preprocessor.py     # Word encoding/decoding utilities
├── hangman_env.py          # Hangman environment and training loop
├── inference.py            # Model inference and evaluation tools
├── ppo_agent.py            # PPO reinforcement learning agent
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Installation

1. **Clone the repository** (or download the files):
```bash
git clone https://github.com/sohammandal1/Hangman-Playing-Agent
cd hangman-ai
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare word dataset**: 
   - Create a text file (e.g., `words_250000_train.txt`) with one word per line
   - Words should contain only alphabetic characters
   - The training script expects this file in the project directory

## Usage

### Training the Model

```bash
python hangman_env.py
```

This will:
- Load the word dataset from `words_250000_train.txt`
- Train the CANINE+PPO agent for 200 episodes
- Save the best performing model as `best_hangman_model.pt`
- Display training progress and word-solving attempts

### Running Inference

```bash
python inference.py
```

This will:
- Load the trained model from `best_hangman_model.pt`
- Evaluate on 1000 random words from the dataset
- Display success rate and individual word results

### Custom Inference

```python
from inference import PPOGuessWrapper

# Load trained model
wrapper = PPOGuessWrapper("best_hangman_model.pt")

# Play a single word
word = "example"
success = infer_single_word(wrapper, word, verbose=True)
print(f"Success: {success}")
```

## Model Architecture

### CANINE Transformer Component
- Uses Google's CANINE-C model for character-level text processing
- Processes partially revealed word states (e.g., `"h_ng_an"`)
- Extracts contextual embeddings from CLS token

### Additional Components
- **Guess Encoder**: Linear layer processing 26-dimensional guess history vector
- **Feature Fusion**: Combines CANINE embeddings with guess encodings
- **Action Head**: Maps to 26 letter probabilities for next guess selection

### PPO Agent
- **Policy Network**: Uses the CANINE model as the policy network
- **Value Function**: Integrated within the same network architecture
- **Training**: Standard PPO with clipped surrogate objective

## Game Environment

### State Representation
- **Word State**: Encoded as sequence of character indices with special tokens for blanks and padding
- **Guess History**: 26-dimensional binary vector indicating previously guessed letters
- **Maximum Length**: 20 characters (configurable)

### Reward Structure
- **Correct Guess**: +1 point per letter revealed
- **Incorrect Guess**: -0.5 points
- **Word Completion**: +5 bonus points
- **Game Failure**: -5 penalty points
- **Repeated Guess**: -1 point

### Game Rules
- Maximum 6 incorrect guesses allowed
- Words contain only lowercase letters
- Case-insensitive gameplay

## Data Processing

### Vocabulary
- **Characters**: a-z (indices 0-25)
- **Special Tokens**: 
  - `_` for unrevealed letters (index 26)
  - `<PAD>` for padding (index 27)

### Encoding Functions
- `encode_word_state()`: Convert word + reveal mask to tensor
- `encode_guesses()`: Convert guess list to binary vector
- `decode_word_state()`: Convert tensor back to readable format

## Training Details

### Hyperparameters
- **Learning Rate**: 3e-4
- **Discount Factor (γ)**: 0.95
- **PPO Clip Ratio**: 0.2
- **Episodes**: 200
- **Max Turns per Game**: 6

### Training Process
1. Random word selection from dataset
2. Environment reset with new word
3. Policy rollout (up to 6 guesses)
4. PPO policy update using collected trajectories
5. Model checkpointing based on episode rewards

## Performance Evaluation

The model evaluation provides:
- **Success Rate**: Percentage of words solved within 6 guesses
- **Individual Results**: Word-by-word success/failure tracking
- **Detailed Logs**: Step-by-step guess progression during inference

## Technical Requirements

### Dependencies
- Python 3.7+
- PyTorch 2.8.0+
- Transformers 4.56.0+
- NumPy, tqdm, and other utilities

### Hardware
- **GPU**: Recommended for faster training (CUDA compatible)
- **Memory**: 8GB+ RAM recommended for large word datasets
- **Storage**: Minimal requirements (~100MB for models and data)

## Customization

### Model Architecture
Modify `canine_model.py` to adjust:
- Hidden dimensions
- Number of classification layers
- CANINE configuration parameters

### Training Parameters
Edit `hangman_env.py` to change:
- Number of training episodes
- Reward structure
- Maximum word length
- Game difficulty settings

### Dataset
- Use any text file with one word per line
- Words should be lowercase and alphabetic only
- Larger datasets generally improve performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size or use CPU-only training
   - Clear GPU cache with `torch.cuda.empty_cache()`

2. **Model Not Learning**:
   - Check word dataset format (one word per line)
   - Verify reward structure in environment
   - Adjust learning rate or training episodes

3. **Import Errors**:
   - Ensure all dependencies are installed correctly
   - Check Python version compatibility

4. **File Not Found**:
   - Verify word dataset path in training script
   - Ensure model checkpoint exists for inference

### Debug Mode
Enable verbose logging by setting `verbose=True` in inference functions or adding print statements in training loops.

## Future Enhancements

- **Multi-word Support**: Extend to phrases and compound words
- **Dynamic Difficulty**: Adaptive word selection based on performance
- **Web Interface**: Create interactive web demo
- **Model Ensemble**: Combine multiple architectures for better performance
- **Advanced RL**: Experiment with other algorithms like A3C or SAC

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Google Research for the CANINE transformer model
- OpenAI for PPO algorithm development
- Hugging Face for transformer implementations
