# deep-poker-rust

Deep learning AI for poker in Rust

This project was a way for me to test new things:

- Rust ownership system & multi-threading
- The candle deep learning library
- Self-play reinforcement learning w/ PPO

It is based mostly on the AlphaHoldem paper, implementing:

- Pseudo-siamese embedding layer with convolution & residual connections
- Actor / critic architecture
- Trinal-clip PPO taken from the paper
- A pool of agents for self-play with ELO rating () a custom multi-player ELO rating
