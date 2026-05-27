# Maze AI Experiment

This project is a Maze AI Experiment that serves as an introduction to AI and reinforcement learning. The application allows users to create, edit, and manage bot profiles that navigate through a maze using different AI strategies, such as Q-Learning. It includes features for training bots, visualizing their learning progress, and customizing their parameters and reward configurations.

## Getting Started

### Prerequisites

- Python 3.10+
- System Tk libraries for Tkinter UI (e.g., `python3-tk` on Debian/Ubuntu)
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/maze-ai-experiment.git
   cd maze-ai-experiment
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   # From the repo root
   python code/main.py
   ```

## Features

- **Profile Management:** Create, edit, and delete profiles for different bots.
- **Bot Training:** Train bots with customizable parameters and reward configurations.
- **Visualizations:** Visualize the bot's learning progress, including reward graphs and heatmaps of visited maze areas.
- **Customization:** Adjust learning rate, discount factor, and rewards for various bot behaviors.

## Project Structure

- `code/ui/app.py`: Main application wiring (Tkinter UI + navigation)
- `code/gameEnvironment.py`: Environment setup and training/maze orchestration
- `code/qLearningBot.py`: Q-Learning agent and episode/visualization flows
- `code/rewardSystem.py`: Reward shaping and event scoring logic
- `code/botProfile.py`: Profile model and persistence helpers
- `code/botConfigs.py`: Bot configuration classes and defaults
- `code/visualizationStrategy.py`: Visualization adapters per bot type
- `code/displayTools.py`: Heatmap and UI utilities
- `code/services/`: Shared services (artifacts repository, runners, training controller)
- `code/ui/frames/`: UI frames for profile management, training, builder, visualization
- `profiles/`: Runtime artifacts (q_tables, rewards, mazes.json) per profile (git-ignored)
- `mazes/`: Saved custom mazes (git-ignored)

## Customization

### Bot Configurations

The bot configurations, including learning rates, discount factors, and rewards, can be customized through the `Create/Edit Profile` interface. Each bot type has specific parameters and rewards that can be adjusted.

### Visualization

The `Visualization` frame provides various visual representations of the bot's learning process. This includes reward graphs, Q-table displays, and heatmaps showing areas visited by the bot.

### Visualization Semantics (Unified)

Live visualization is strictly inference-only across all bots. When the Visualization window is open for a profile, training for that profile is paused and visualization steps do not modify training artifacts.

- No learning: No Q-table updates, backprop, or optimizer steps during visualization.
- No persistence: No writes to `profiles/<name>/mazes.json`, `SimulationRewards.txt`, or weights.
- No training counters: Do not advance training/global steps or fill replay buffers.
- In-memory only: Bot position and the in-memory heatmap used for drawing update normally.
- Episode lifecycle: `begin_visualization_episode()` initializes state, `step_visualization(max_steps)` advances up to N steps and returns True when the episode ends, and a finalize step resets only local visualization state.

## Usage

1. **Create a Profile:** Navigate to `Profile Management` and create a new profile.
2. **Set Parameters:** Customize the bot's parameters and reward configurations.
3. **Train the Bot:** Go to `Bot Training` and select the profile to start training.
4. **Visualize Progress:** Check the `Visualizations` section to see how the bot is learning.

### Headless CLI (optional)

Run a profile for N episodes without the GUI and print summary metrics:

```bash
python code/debugCli.py --profile TEST --episodes 50 --save-json
```

Use `--create` to create a profile first (QLearningBot only):

```bash
python code/debugCli.py --profile TEST --create --episodes 10
```

## Type Checking

Run strict typing checks with Pyright from the project virtual environment:

```bash
./.venv/Scripts/pyright.exe
```

If Pyright is not installed in the environment yet:

```bash
python -m pip install pyright
./.venv/Scripts/pyright.exe
```

CI should use the same command (`.venv/Scripts/pyright.exe`) so local and CI reports stay aligned.

## Notes

Folders `profiles/` and `mazes/` are created automatically at runtime. They are git-ignored by default. This project is a learning exercise in reinforcement learning and AI-driven UI applications.
