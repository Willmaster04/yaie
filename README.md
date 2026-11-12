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

## Demo API

Run the API server with:

```sh
cargo run --bin api
```

By default, it listens on `http://127.0.0.1:3000`. Check the demo endpoints:

```sh
curl http://127.0.0.1:3000/
curl http://127.0.0.1:3000/health
```

Set `YAIE_API_ADDR` to change the bind address, for example
`YAIE_API_ADDR=127.0.0.1:8080 cargo run --bin api`.

These are demo endpoints; inference is not yet connected to the API.