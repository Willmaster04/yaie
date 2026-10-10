use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("HOME")?).join("models/tinyllama");
    let device = Device::Cpu;

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!(e))?;
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let config = config.into_config(false); // false = no flash-attn

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)?
    };
    let model = Llama::load(vb, &config)?;
    let mut cache = Cache::new(true, DType::F32, &config, &device)?;

    let prompt = "The capital of France is";
    let mut tokens: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow!(e))?
        .get_ids()
        .to_vec();

    let mut index_pos = 0;
    for step in 0..50 {
        // Prefill: whole prompt on step 0. Decode: just the last token after that.
        let ctx = if step == 0 { &tokens[..] } else { &tokens[tokens.len() - 1..] };
        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos, &mut cache)?.squeeze(0)?;
        index_pos += ctx.len();

        let next = logits.argmax(0)?.to_scalar::<u32>()?;
        if next == 2 { break; } // TinyLlama EOS
        tokens.push(next);
    }

    println!("{}", tokenizer.decode(&tokens, true).map_err(|e| anyhow!(e))?);
    Ok(())
}