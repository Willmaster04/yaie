use std::env;

use yaie::api::routes::router;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let address = env::var("YAIE_API_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".to_owned());
    let listener = tokio::net::TcpListener::bind(&address).await?;

    println!("YAIE demo API listening on http://{address}");
    axum::serve(listener, router()).await?;
    Ok(())
}
