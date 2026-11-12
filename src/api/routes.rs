use axum::{Json, Router, routing::get};
use serde_json::{Value, json};

pub fn router() -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
}

async fn index() -> Json<Value> {
    Json(json!({
        "service": "yaie",
        "message": "Hello, World!"
    }))
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}
