# Deep learning AI for poker in Rust

This project was a way for me to test new things:
- Rust ownership system & multi-threading
- The [candle](https://github.com/huggingface/candle) deep learning library
- Self-play reinforcement learning w/ PPO

It ended up as a quite complete project. I did not want to use any existing reinforcement learning library like OpenAI's Gymnasium in order to make sure I can tweak the whole process, and to build a Rust-only project.

## Architecture

It is based mostly on the [AlphaHoldem paper](https://ojs.aaai.org/index.php/AAAI/article/view/20394) with adaptations for more than 2 players, implementing:
- Pseudo-siamese embedding layer with convolution & residual connections
- Actor / critic architecture
- Trinal-clip PPO
- A pool of agents for self-play with ELO rating
- Use of epsilon-greedy algorithm and entropy to favor exploration

## Experiments

I ran it on a 32-core CPU and a 8GB RTX 2070 GPU also used for display, so the model is not very big and would need to be larger to reach the same level of performance as AlphaHoldem. But it still converges to a valid strategy.

My conclusion is that there are two very important factors to have good results with this approach:

- The training hardware, so obviously the model size. AlphaHoldem was trained on 8x NVIDIA Titan V GPUs. I had to move rollout & tournament updates to CPU to improve performance on my computer. The current batch size and learning rate are also much lower on my hardware, so it is slow to converge.

- How agents are selected for self-play. A wrong selection of agents can lead to very exotic strategies. I also believe that having some agents pretrained with hand histories, or on CFR implementations would help faster convergence. Having different pools of agents with different strategies would help too, like the [AlphaStar AI](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/).