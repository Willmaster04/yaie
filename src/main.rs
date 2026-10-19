fn main() -> Result<()> {
    // load the model from models folder

    // load weights into memory

    // init KV cache

    // init some prompt
    // tokenize the prompt using tokenizer.rs

    // loop

    // prefill initially
    // get logits from executor.rs
    // argmax for next token

    // append next token to the prompt
    // repear until EOS or max length reached

    // decode tokens back to text
    // using tokenizer.rs
    
    // return output to caller
    Ok(())
}