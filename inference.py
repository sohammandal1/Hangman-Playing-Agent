import torch
from canine_model import HangmanCANINE
from ppo_agent import PPOAgent
from data_preprocessor import encode_word_state, encode_guesses, load_word_dataset, IDX2CHAR, encode_word_state_from_revealed
from typing import List
from hangman_env import HangmanEnv
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOGuessWrapper:
    def __init__(self, model_path="best_hangman_model.pt"):
        self.model = HangmanCANINE().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.env = None

    def reset_env(self, word):
        from hangman_env import HangmanEnv
        self.env = HangmanEnv(word)

    def get_next_guess(self, revealed_word: str, guessed_letters: List[str]) -> str:
        with torch.no_grad():
            word_tensor = encode_word_state_from_revealed(
                revealed_word).unsqueeze(0).to(device)
            guess_tensor = encode_guesses(
                guessed_letters).unsqueeze(0).to(device)
            logits = self.model(word_tensor, guess_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return IDX2CHAR[action.item()]


def infer_single_word(wrapper: PPOGuessWrapper, word: str, verbose=True, max_steps=6):
    wrapper.reset_env(word)
    env = wrapper.env
    guessed_letters = []

    for t in range(max_steps):
        revealed_word = ''.join(c if r else '_' for c,
                                r in zip(env.word, env.revealed))
        guess = wrapper.get_next_guess(revealed_word, guessed_letters)
        guessed_letters.append(guess)
        reward, done = env.step(guess)

        revealed_state = ''.join(
            c if r else '_' for c, r in zip(env.word, env.revealed))
        if verbose:
            print(
                f"Infer Step {t + 1}: Guessed '{guess}' | Reward: {reward} | Revealed: {revealed_state}")

        if done:
            if verbose:
                print(f"🎉 Word '{word}' completed in {t + 1} steps!\n")
            return 1  # success

    if verbose:
        print(f"❌ Word '{word}' not fully revealed in {max_steps} steps.\n")
    return 0  # fail


def evaluate_model(words: List[str], model_path="best_hangman_model.pt"):
    wrapper = PPOGuessWrapper(model_path=model_path)
    print(f"✅ Loaded model from {model_path}")
    print(f"📋 Starting evaluation on {len(words)} words...\n")

    scores = []
    for word in words:
        score = infer_single_word(wrapper, word, verbose=False)
        print(f"Word: {word} | Score: {score}")
        scores.append(score)

    success_rate = sum(scores) / len(scores)
    print(f"\n✅ Evaluation Complete!")
    print(f"✅ Total Words: {len(words)} | Success: {sum(scores)}")
    print(f"📈 Success Rate: {success_rate * 100:.2f}%")

    return scores

if __name__ == '__main__':
    total_word_list = load_word_dataset("words_250000_train.txt")
    word_list = random.sample(total_word_list, 1000)  
    evaluate_model(word_list)
