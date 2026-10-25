use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaEosToks};

fn main() -> anyhow::Result<()> {
    let model_name = "tinyllama";
    let model_path = Path::new(&get_home_dir()).join("models").join(model_name);
    let device = Device::Cpu;

    // load the model from models folder
    let config_file = fs::File::open(model_path.join("config.json"))?;
    let llama_config: candle_transformers::models::llama::LlamaConfig =
        serde_json::from_reader(config_file)?;
    let config = llama_config.into_config(false);

    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let mut weight_paths = fs::read_dir(&model_path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    weight_paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "safetensors")
    });
    weight_paths.sort();
    if weight_paths.is_empty() {
        anyhow::bail!(
            "No .safetensors model weights found in {}",
            model_path.display()
        );
    }

    let var_builder =
        unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, DType::F32, &device)? };
    let model = Llama::load(var_builder, &config)?;

    let mut cache = Cache::new(true, DType::F32, &config, &device)?;

    let prompt = get_prompt();
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?;
    let mut tokens = encoding.get_ids().to_vec();
    if tokens.is_empty() {
        anyhow::bail!("The prompt produced no tokens");
    }
    if tokens.len() > config.max_position_embeddings {
        anyhow::bail!("The prompt is longer than the model's context length");
    }

    let prompt_length = tokens.len();
    let mut position = 0;

    // inference loop
    for n in 0..100 {
        let context = if n == 0 {
            tokens.as_slice()
        } else {
            &tokens[tokens.len() - 1..]
        };

        let context_length = context.len();
        // get logits from the model
        let input = Tensor::new(context, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, position, &mut cache)?;

        // argmax for next token
        let next_token = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;
        let is_eos = match &config.eos_token_id {
            Some(LlamaEosToks::Single(eos)) => next_token == *eos,
            Some(LlamaEosToks::Multiple(eos_tokens)) => eos_tokens.contains(&next_token),
            None => false,
        };
        if is_eos {
            break;
        }

        // append next token to the prompt
        tokens.push(next_token);
        position += context_length;
        if position >= config.max_position_embeddings {
            break;
        }
    }

    // decode tokens back to text
    let output = tokenizer
        .decode(&tokens[prompt_length..], true)
        .map_err(anyhow::Error::msg)?;

    println!("{output}");
    Ok(())
}

fn get_prompt() -> String {
    println!("Please enter a prompt:");
    let mut prompt = String::new();
    match io::stdin().read_line(&mut prompt) {
        Ok(_) => prompt,
        Err(error) => panic!("Error: {error}"),
    }
}

fn get_home_dir() -> PathBuf {
    match env::home_dir() {
        Some(path) => path,
        None => panic!("Impossible to get your home dir!"),
    }
}
