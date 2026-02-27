# Exercise 1.12: PPO Implementation for MountainCar-v0

This repository provides the starter code for Exercise 1.12. Your task is to implement the **Proximal Policy Optimization (PPO)** algorithm to solve the classic `MountainCar-v0` reinforcement learning task.

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `PPO_MountainCar-v0.py` and complete the `# YOUR CODE HERE` block inside the `update` method. You need to:<br>  • Calculate the importance ratio.<br>  • Update the actor network.<br>  • Update the critic network. |
| **Run** | Execute: `python PPO_MountainCar-v0.py`<br>*(Note: Ensure you have installed the `gym` and `tensorboardX` libraries first).* |
| **Observe** | The training steps and metrics are logged via TensorBoard to the `../exp` directory. Observe how the agent learns to build momentum to reach the flag. |



## Files
| File | Purpose |
|------|---------|
| `PPO_MountainCar-v0.py` | Starter script (with TODO). |