use crate::error::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct RecognitionRequest {
    pub fingerprint: String,
}

#[derive(Serialize, Deserialize)]
pub struct RecognitionResponse {
    pub matched: bool,
    pub confidence: f32,
    pub track_name: Option<String>,
    pub artist: Option<String>,
}

pub async fn recognize(
    fingerprint: &str,
    api_url: &str,
) -> Result<RecognitionResponse> {
    let client = reqwest::Client::new();
    
    let response = client
        .post(format!("{}/api/recognize", api_url))
        .json(&RecognitionRequest {
            fingerprint: fingerprint.to_string(),
        })
        .send()
        .await?
        .json::<RecognitionResponse>()
        .await?;
    
    Ok(response)
}
