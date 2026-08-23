# Recording Analysis API Documentation

## Overview

The Recording Analysis API analyzes sales call recordings using AI (Gemini) to extract structured insights including summaries, sentiment analysis, keywords, action items, and more. The API supports both Bangla and English output languages.

**Base URL:** `http://localhost:8000` (or your deployed server URL)

**API Version:** 1.0.0

---

## Endpoints

### 1. Health Check

**`GET /health`**

Returns the health status of the API.

**Response:**

```json
{
  "status": "ok"
}
```

**Status Code:** `200 OK`

---

### 2. Analyze Call Recording

**`POST /crm/analyze-call`**

Analyzes a sales call recording and returns structured analysis including summary, sentiment, keywords, action items, and more.

**Content-Type:** `multipart/form-data`

#### Request Parameters

| Parameter              | Type          | Required | Description                                                                                                              |
| ---------------------- | ------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `audio_file`         | File (binary) | Yes      | The call recording file. Supported formats: mp3, wav, ogg, m4a, and other common audio formats.**Max size: 200MB** |
| `language`           | String (enum) | Yes      | Output language for the analysis. Valid values:`"bn"` (Bangla) or `"en"` (English)                                   |
| `agent_name`         | String        | No       | Name of the sales agent on the call                                                                                      |
| `customer_name`      | String        | No       | Name of the customer on the call                                                                                         |
| `call_direction`     | String (enum) | No       | Direction of the call. Valid values:`"outbound"` or `"inbound"`                                                      |
| `deal_or_lead_id`    | String        | No       | CRM deal or lead reference ID                                                                                            |
| `product_or_service` | String        | No       | Product or service this call relates to                                                                                  |

#### Example Request

```bash
curl -X POST "http://localhost:8000/crm/analyze-call" \
  -F "audio_file=@call_recording.mp3" \
  -F "language=en" \
  -F "agent_name=John Smith" \
  -F "customer_name=Jane Doe" \
  -F "call_direction=outbound" \
  -F "deal_or_lead_id=DEAL-12345" \
  -F "product_or_service=Enterprise CRM Suite"
```

#### Success Response

**Status Code:** `200 OK`

```json
{
  "language": "en",
  "analysis": {
    "summary": "The agent introduced the CRM suite and discussed pricing tiers with the customer. The customer expressed interest in the Pro plan but raised concerns about integration with their existing ERP system. The agent assured them that integration support is included and offered to schedule a demo. The customer agreed to a follow-up demo scheduled for next Tuesday.",
    "keywords": [
      "CRM suite",
      "Pro plan",
      "ERP integration",
      "pricing",
      "demo",
      "enterprise",
      "API access",
      "onboarding"
    ],
    "customer_sentiment": "positive",
    "call_outcome": "Demo scheduled",
    "customer_pain_points": [
      "Integration with existing ERP system",
      "Need for bulk data migration"
    ],
    "objections_raised": [
      "Concerned about ERP integration complexity"
    ],
    "products_services_discussed": [
      "Enterprise CRM Suite",
      "Pro Plan ($499/month)",
      "Integration Support Package"
    ],
    "action_items": [
      "Agent to send demo link to customer",
      "Customer to share ERP documentation for integration assessment",
      "Follow-up call scheduled for Tuesday 2:00 PM"
    ],
    "follow_up_required": true,
    "follow_up_notes": "Schedule demo for Tuesday 2:00 PM. Prepare integration assessment based on ERP documentation from customer.",
    "important_notes": "Customer mentioned they are currently using CompetitorX and are looking to switch due to pricing concerns."
  },
  "token_usage": {
    "input_tokens": 15234,
    "output_tokens": 856,
    "total_tokens": 16090,
    "audio_tokens": 12500
  }
}
```

#### Response Schema

| Field           | Type                          | Description                               |
| --------------- | ----------------------------- | ----------------------------------------- |
| `language`    | String (`"bn"` or `"en"`) | The output language used for the analysis |
| `analysis`    | Object                        | The structured call analysis (see below)  |
| `token_usage` | Object or null                | Token usage statistics for the request    |

##### `analysis` Object

| Field                           | Type             | Description                                                                                                                            |
| ------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `summary`                     | String           | A concise but complete narrative summary of the call (4-8 sentences)                                                                   |
| `keywords`                    | Array of Strings | 8-15 important keywords/short phrases from the call (products, topics, objections, competitors, etc.)                                  |
| `customer_sentiment`          | String           | One of:`"positive"`, `"neutral"`, `"negative"`, `"mixed"`                                                                      |
| `call_outcome`                | String           | Short label for how the call ended (e.g., "Demo scheduled", "Not interested", "Needs follow-up", "Deal closed", "No answer/voicemail") |
| `customer_pain_points`        | Array of Strings | Problems, needs, or goals the customer expressed                                                                                       |
| `objections_raised`           | Array of Strings | Objections, hesitations, or concerns raised by the customer                                                                            |
| `products_services_discussed` | Array of Strings | Products, services, plans, or prices discussed on the call                                                                             |
| `action_items`                | Array of Strings | Concrete action items or commitments made by either side                                                                               |
| `follow_up_required`          | Boolean          | Whether a follow-up call or action is needed                                                                                           |
| `follow_up_notes`             | String or null   | When/what the follow-up should be about                                                                                                |
| `important_notes`             | String or null   | Anything else important for the sales rep                                                                                              |

##### `token_usage` Object

| Field             | Type            | Description                                              |
| ----------------- | --------------- | -------------------------------------------------------- |
| `input_tokens`  | Integer         | Tokens consumed by the prompt + audio input              |
| `output_tokens` | Integer         | Tokens generated in the response                         |
| `total_tokens`  | Integer         | input_tokens + output_tokens                             |
| `audio_tokens`  | Integer or null | Portion of input_tokens attributable to the audio itself |

---

## Error Responses

### 400 Bad Request

Returned when the uploaded file is not a valid audio file or is empty.

```json
{
  "detail": "Uploaded file does not look like an audio file."
}
```

```json
{
  "detail": "Uploaded audio file is empty."
}
```

### 502 Bad Gateway

Returned when the AI service (Gemini) fails to process the request.

```json
{
  "detail": "Audio file exceeds the 200MB limit."
}
```

```json
{
  "detail": "Timed out waiting for Gemini to process the audio file."
}
```

```json
{
  "detail": "Model response could not be parsed into the expected schema: [error details]"
}
```

### 500 Internal Server Error

Returned for unexpected server errors.

```json
{
  "detail": "Unexpected error while analyzing the call."
}
```

---

## Implementation Notes

### Processing Flow

1. **Upload Audio** - The API uploads the audio file to Google's Gemini Files API
2. **Wait for Processing** - Polls until Google's servers finish processing the audio (timeout: 120 seconds, poll interval: 2 seconds)
3. **Analyze** - Sends the audio to Gemini with structured output schema
4. **Cleanup** - Deletes the temporary file from Google's servers
5. **Return** - Returns the structured analysis

### Language Support

- **Input Audio:** Supports Bangla, English, or Bangla-English code-mixed speech
- **Output Language:** All text fields in the response are written in the requested output language (`"bn"` for Bangla, `"en"` for English)
- **Proper Nouns:** Person names, company names, brand/product names are kept as-is (not translated)
- **Numbers:** Currency amounts, phone numbers, dates are always in standard numerals (e.g., "1200 BDT", "25 August")

### Constraints

- **Max Audio Size:** 200MB
- **Processing Timeout:** 120 seconds
- **Supported Audio Formats:** mp3, wav, ogg, m4a, and other common audio formats

---

## Example: Python Integration

```python
import requests

def analyze_call(audio_file_path: str, language: str = "en", **kwargs):
    """Analyze a call recording using the Recording Analysis API."""
  
    url = "http://localhost:8000/crm/analyze-call"
  
    with open(audio_file_path, "rb") as f:
        files = {"audio_file": f}
        data = {"language": language}
        data.update(kwargs)
      
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()

# Usage
result = analyze_call(
    audio_file_path="call_recording.mp3",
    language="en",
    agent_name="John Smith",
    customer_name="Jane Doe",
    call_direction="outbound",
    deal_or_lead_id="DEAL-12345",
    product_or_service="Enterprise CRM Suite"
)

print(result["analysis"]["summary"])
print(f"Sentiment: {result['analysis']['customer_sentiment']}")
print(f"Follow-up required: {result['analysis']['follow_up_required']}")
```

---

## Example: JavaScript/Node.js Integration

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function analyzeCall(audioFilePath, language = 'en', context = {}) {
  const form = new FormData();
  form.append('audio_file', fs.createReadStream(audioFilePath));
  form.append('language', language);
  
  // Add optional context
  Object.entries(context).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      form.append(key, value);
    }
  });
  
  const response = await axios.post(
    'http://localhost:8000/crm/analyze-call',
    form,
    { headers: form.getHeaders() }
  );
  
  return response.data;
}

// Usage
analyzeCall('call_recording.mp3', 'en', {
  agent_name: 'John Smith',
  customer_name: 'Jane Doe',
  call_direction: 'outbound',
  deal_or_lead_id: 'DEAL-12345'
}).then(result => {
  console.log(result.analysis.summary);
  console.log('Sentiment:', result.analysis.customer_sentiment);
});
```

---

## Environment Variables

The API requires the following environment variables:

| Variable                              | Required | Default                   | Description                          |
| ------------------------------------- | -------- | ------------------------- | ------------------------------------ |
| `GOOGLE_API_KEY`                    | Yes      | -                         | Google API key for Gemini access     |
| `GEMINI_MODEL`                      | No       | `gemini-3.1-flash-lite` | Gemini model to use                  |
| `FILE_PROCESSING_POLL_INTERVAL_SEC` | No       | `2.0`                   | Polling interval for file processing |
| `FILE_PROCESSING_TIMEOUT_SEC`       | No       | `120.0`                 | Timeout for file processing          |
| `MAX_AUDIO_SIZE_MB`                 | No       | `200`                   | Maximum audio file size in MB        |
