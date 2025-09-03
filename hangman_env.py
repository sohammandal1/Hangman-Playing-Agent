import torch
from data_preprocessor import encode_word_state, encode_guesses, load_word_dataset
from canine_model import HangmanCANINE
from ppo_agent import PPOAgent
import random


class HangmanEnv:
    def __init__(self, word, max_turns=6):
        self.word = word
        self.guessed = []
        self.revealed = [False] * len(word)
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False

    def get_state(self):
        return encode_word_state(self.word, self.revealed), encode_guesses(self.guessed)

    def step(self, guess):
        if self.done:
            return 0.0, True  # No more steps allowed

        self.current_turn += 1

        if guess in self.guessed:
            return -1.0, self._check_done()

        self.guessed.append(guess)
        reward = 0
        found = False
        occurrences = 0

        for i, ch in enumerate(self.word):
            if ch == guess and not self.revealed[i]:
                self.revealed[i] = True
                found = True
                occurrences += 1

        if found:
            reward += occurrences
        else:
            reward -= 0.5

        if all(self.revealed):
            reward += 5.0
            self.done = True
        elif self.current_turn >= self.max_turns:
            reward -= 5.0  # penalty for failing to solve within allowed turns
            self.done = True

        return reward, self.done

    def _check_done(self):
        return self.done or self.current_turn >= self.max_turns


def train(save_path="best_hangman_model.pt", word_dataset_path="words_250000_train.txt"):
    model = HangmanCANINE()
    agent = PPOAgent(model)
    best_total_reward = float('-inf')
    words = load_word_dataset(word_dataset_path)

    for episode in range(200):
        word = random.choice(words)
        env = HangmanEnv(word)
        log_probs = []
        values = []
        rewards = []
        masks = []
        actions = []

        print(f"\n=== Starting Episode {episode} ===")
        print(f"Target word: {word} (length: {len(env.word)})")

        for t in range(6):
            word_tensor, guess_tensor = env.get_state()
            word_tensor = word_tensor.unsqueeze(0)
            guess_tensor = guess_tensor.unsqueeze(0)
            logits = model(word_tensor, guess_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            guess = chr(ord('a') + action.item())
            reward, done = env.step(guess)

            log_probs.append(dist.log_prob(action))
            values.append(logits.max(dim=1).values)
            rewards.append(reward)
            masks.append(1 - int(done))
            actions.append(action)

            revealed_state = ''.join(
                c if r else '_' for c, r in zip(env.word, env.revealed)
            )
            print(
                f"Step {t + 1}: Guessed '{guess}' | Reward: {reward} | Revealed: {revealed_state}")

            if done:
                print(f"Word fully revealed in {t + 1} steps!")
                break

            if not done and t == 5:
                print("❌ Max attempts reached. Word not solved!")

        agent.update(log_probs, values, rewards, masks, actions)
        print(f"=== Episode {episode} completed ===")

        total_reward = sum(rewards)
        print(f"Episode {episode} total reward: {total_reward:.2f}")

        # Save model if performance improves
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved with reward: {total_reward:.2f}")

    print(
        f"Training Completed! Best model saved with reward: {best_total_reward:.2f} ✅")

if __name__ == '__main__':
    train(save_path="best_hangman_model.pt", word_dataset_path="words_250000_train.txt")
