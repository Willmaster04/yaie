Yet Another Inference Engine
-- pronounced yay

A rust based inference engine

Tokenizer 
↓
Scheduler ⇄ KV Cache Manager
    ↓
Model Executor
    ↓
Sampler
    ↓
Detokenizer / stop checks → Stream to client
    ↑___________________|
(unfinished sequences loop back)