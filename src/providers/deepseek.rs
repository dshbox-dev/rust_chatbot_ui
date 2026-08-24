use super::{ProviderError, ProviderTrait, BaseProvider};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use std::fmt;
use std::sync::{Arc, Mutex};
use crate::app::ProfileType;
use futures_util::StreamExt;

pub struct DeepSeek {
    base: Arc<Mutex<BaseProvider>>,
    current_model: Arc<Mutex<String>>,
}

impl DeepSeek {
    pub fn new(api_key: String) -> Self {
        Self {
            base: Arc::new(Mutex::new(BaseProvider::new(api_key))),
            current_model: Arc::new(Mutex::new("deepseek-v4-flash".to_string())),
        }
    }
}

impl ProviderTrait for DeepSeek {
    fn name(&self) -> &'static str {
        "DeepSeek"
    }

    fn models(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("deepseek-v4-flash", 1048576),
            ("deepseek-v4-pro", 1048576),
            ("deepseek-v4-flash-vision-exp", 1048576),
        ]
    }

    fn stream_response(&self, messages: Vec<Value>) -> Result<mpsc::Receiver<String>, ProviderError> {
        let model = self.current_model.lock().unwrap().clone();
        let (top_p, _top_k, _repetition_penalty, creativity) = self.base.lock().unwrap().get_parameters();
        let client = self.base.lock().unwrap().get_client();
        let api_key = self.base.lock().unwrap().get_api_key();

        // DeepSeek uses an OpenAI-compatible chat completions API.
        // frequency_penalty / presence_penalty are deprecated by DeepSeek and are omitted.
        let json_body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "max_tokens": 16384,
            "temperature": creativity,
            "top_p": top_p,
        });

        println!("Debug: Model parameters - top_p: {}, creativity: {}", top_p, creativity);

        let (tx, rx) = mpsc::channel(1024);

        tokio::task::spawn(async move {
            let response = match client
                .post("https://api.deepseek.com/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&json_body)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    let _ = tx.send(format!("Error: {}", ProviderError::RequestError(e.to_string()))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let error_body = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let _ = tx.send(format!("Error: {}", ProviderError::ResponseError(error_body))).await;
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(item) = stream.next().await {
                match item {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();

                            if line.starts_with("data: ") {
                                let data = &line["data: ".len()..];
                                if data == "[DONE]" {
                                    return;
                                }
                                if let Ok(json) = serde_json::from_str::<Value>(data) {
                                    let delta = &json["choices"][0]["delta"];
                                    // In thinking mode, reasoning deltas arrive before the answer.
                                    if let Some(reasoning) = delta["reasoning_content"].as_str() {
                                        if tx.send(reasoning.to_string()).await.is_err() {
                                            return;
                                        }
                                    }
                                    if let Some(content) = delta["content"].as_str() {
                                        if tx.send(content.to_string()).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(format!("Error: {}", ProviderError::StreamError(e.to_string()))).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }

    fn set_current_model(&self, model: String) {
        *self.current_model.lock().unwrap() = model;
    }

    fn update_profile(&self, profile: ProfileType) {
        self.base.lock().unwrap().update_profile(profile);
    }

    fn get_parameters(&self) -> (f32, u32, f32, f32) {
        self.base.lock().unwrap().get_parameters()
    }
}

impl fmt::Display for DeepSeek {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeepSeek")
    }
}
