# API Documentation

## Emotion-Aware AI Companion REST API v2.0

Complete API reference for the Emotion-Aware AI Companion backend.

---

## Base URL

```
Production: https://api.yourdomain.com
Development: http://localhost:8000
```

---

## Authentication

Most endpoints require authentication using JWT tokens.

### Get Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGci...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGci...
```

---

## Health & Status

### Health Check

Get system health status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "services": {
    "sentiment_analyzer": "running",
    "fusion_engine": "running",
    "adaptive_system": "running"
  }
}
```

### API Status

Get detailed API status and metrics.

```http
GET /api/v1/status
```

**Response:**
```json
{
  "status": "running",
  "active_sessions": 42,
  "uptime": "5d 12h 30m",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Emotion Detection

### Detect Emotions (Multimodal)

Analyze emotions from multiple input modalities.

```http
POST /api/v1/emotion/detect
Content-Type: application/json

{
  "text": "I'm feeling great today!",
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "audio_base64": "SUQzBAAAAAAAI1RTU0...",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "fused_emotion": {
    "dominant_emotion": "happy",
    "emotions": {
      "happy": 0.85,
      "neutral": 0.10,
      "surprised": 0.05
    },
    "confidence": 0.85,
    "modalities_used": ["text", "face", "voice"],
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "individual_emotions": {
    "text": {...},
    "face": {...},
    "voice": {...}
  }
}
```

### Analyze Text Sentiment

Analyze sentiment from text only.

```http
POST /api/v1/emotion/text?text=I'm%20happy&user_id=user123
```

**Response:**
```json
{
  "sentiment": "positive",
  "sentiment_scores": {
    "positive": 0.8,
    "negative": 0.05,
    "neutral": 0.15,
    "compound": 0.75
  },
  "dominant_emotion": "happy",
  "emotions": {
    "happy": 0.8,
    "neutral": 0.2
  },
  "confidence": 0.8,
  "text_features": {
    "polarity": 0.75,
    "subjectivity": 0.6,
    "word_count": 3,
    "char_count": 9
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Chat & Conversation

### Send Chat Message

Send a message and get an adaptive response.

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "I'm feeling stressed about work",
  "user_id": "user123",
  "emotion_context": {
    "dominant_emotion": "stressed",
    "confidence": 0.75
  },
  "include_voice": false
}
```

**Response:**
```json
{
  "response": "I understand you're feeling stressed about work. That's completely valid. Would you like to try a quick breathing exercise, or would you prefer to talk through what's on your mind?",
  "emotion_context": {
    "dominant_emotion": "stressed",
    "confidence": 0.75,
    "emotions": {...}
  },
  "response_mode": "supportive",
  "confidence": 0.75,
  "audio_base64": null,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Stream Chat Response

Get streaming response for real-time updates.

```http
POST /api/v1/chat/stream
Content-Type: application/json
Accept: text/event-stream

{
  "message": "Hello!",
  "user_id": "user123"
}
```

**Response (SSE):**
```
data: {"type": "token", "content": "Hello"}
data: {"type": "token", "content": "!"}
data: {"type": "token", "content": " How"}
data: {"type": "done", "full_response": "Hello! How can I help?"}
```

---

## User Profile & Analytics

### Get User Statistics

Get user statistics and analytics.

```http
GET /api/v1/user/user123/stats?days=7
```

**Response:**
```json
{
  "user_id": "user123",
  "total_interactions": 156,
  "mood_patterns": {
    "dominant_emotions": {
      "happy": 45,
      "neutral": 30,
      "sad": 15
    },
    "avg_confidence": 0.72,
    "total_entries": 90,
    "time_patterns": {
      "morning": "happy",
      "afternoon": "neutral",
      "evening": "calm"
    }
  },
  "interaction_stats": {
    "total_interactions": 156,
    "response_modes": {
      "supportive": 40,
      "neutral": 60,
      "cheerful": 56
    },
    "avg_message_length": 85
  }
}
```

### Get Emotion History

Get user's emotion history.

```http
GET /api/v1/user/user123/history?days=7&limit=50
```

**Response:**
```json
{
  "user_id": "user123",
  "history": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "dominant_emotion": "happy",
      "confidence": 0.85,
      "emotions": {...},
      "modalities": ["text", "face"]
    },
    ...
  ],
  "count": 50
}
```

### Set User Preference

Set a user preference.

```http
POST /api/v1/user/user123/preference?key=theme&value=dark
```

**Response:**
```json
{
  "status": "success",
  "key": "theme",
  "value": "dark"
}
```

### Export User Data

Export all user data.

```http
GET /api/v1/user/user123/export
```

**Response:**
```json
{
  "status": "success",
  "filepath": "/path/to/export_user123_20240115.json",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Recommendations

### Get Recommendations

Get personalized recommendations based on emotional state.

```http
POST /api/v1/recommendations
Content-Type: application/json

{
  "emotion_context": {
    "dominant_emotion": "stressed",
    "confidence": 0.8
  },
  "user_id": "user123",
  "limit": 5
}
```

**Response:**
```json
{
  "emotion": "stressed",
  "confidence": 0.8,
  "recommended_for": "stressed",
  "activities": [
    "Take a 10-minute break",
    "Practice mindfulness meditation",
    "Go for a quick walk"
  ],
  "music": ["calm", "ambient", "classical"],
  "exercises": [
    "Box breathing (4-4-4-4)",
    "Body scan meditation"
  ],
  "quick_actions": [
    "Take a break",
    "Start meditation",
    "Go for a walk"
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get Breathing Exercise

Get a breathing exercise for current emotion.

```http
GET /api/v1/recommendations/breathing?emotion=anxious
```

**Response:**
```json
{
  "name": "4-7-8 Breathing",
  "description": "Reduce anxiety with extended exhale",
  "steps": [
    "Inhale through nose for 4 counts",
    "Hold breath for 7 counts",
    "Exhale through mouth for 8 counts",
    "Repeat 4 times"
  ],
  "duration": "2 minutes"
}
```

---

## WebSocket API

### Connect to WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session123');

ws.onopen = () => {
  console.log('Connected to WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Send chat message
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    message: 'Hello!',
    user_id: 'user123'
  }
}));

// Send ping/heartbeat
ws.send(JSON.stringify({type: 'ping'}));

// Send emotion update
ws.send(JSON.stringify({
  type: 'emotion_update',
  data: {
    dominant_emotion: 'happy',
    confidence: 0.85
  }
}));
```

### WebSocket Message Types

**Client → Server:**
- `chat`: Send chat message
- `ping`: Heartbeat
- `emotion_update`: Update emotion state

**Server → Client:**
- `chat_response`: Chat response
- `pong`: Heartbeat response
- `emotion_acknowledged`: Emotion update confirmed
- `error`: Error message

---

## Analytics & Insights

### Get System Analytics

Get system-wide analytics.

```http
GET /api/v1/analytics/summary?days=7
```

**Response:**
```json
{
  "total_sessions": 1234,
  "active_connections": 42,
  "period_days": 7,
  "total_requests": 50000,
  "avg_response_time_ms": 150,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get User Insights

Get AI-generated insights from user patterns.

```http
GET /api/v1/analytics/insights/user123?days=30
```

**Response:**
```json
{
  "insights": [
    {
      "type": "pattern",
      "severity": "medium",
      "title": "Afternoon Mood Pattern",
      "description": "You tend to feel stressed in the afternoon",
      "suggestion": "Consider scheduling breaks during afternoon time"
    },
    {
      "type": "milestone",
      "severity": "low",
      "title": "Active Engagement",
      "description": "You've had 156 meaningful conversations",
      "suggestion": "Great job staying connected with yourself!"
    }
  ]
}
```

---

## Configuration

### Get Public Configuration

Get public configuration information.

```http
GET /api/v1/config
```

**Response:**
```json
{
  "emotion_detection": {
    "modalities": ["face", "voice", "text"],
    "fusion_strategy": "weighted_average"
  },
  "response_modes": [
    "supportive",
    "cheerful",
    "calm",
    "concise",
    "neutral"
  ],
  "features": {
    "voice_cloning_tts": false,
    "music_recommendations": true,
    "break_reminders": true
  }
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

---

## Rate Limiting

API requests are rate-limited to prevent abuse:

- **Authenticated requests**: 60 requests/minute
- **Unauthenticated requests**: 20 requests/minute
- **WebSocket connections**: 10 concurrent connections per user

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248000
```

---

## Pagination

List endpoints support pagination:

```http
GET /api/v1/user/user123/history?page=1&per_page=20
```

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "pages": 5
  }
}
```

---

## Versioning

API version is included in the URL: `/api/v1/...`

Breaking changes will result in a new version: `/api/v2/...`

---

## SDK Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Analyze emotion
response = requests.post(
    f"{BASE_URL}/api/v1/emotion/text",
    params={"text": "I'm happy!", "user_id": "user123"}
)
emotion = response.json()
print(f"Emotion: {emotion['dominant_emotion']}")

# Chat
response = requests.post(
    f"{BASE_URL}/api/v1/chat",
    json={
        "message": "Hello!",
        "user_id": "user123"
    }
)
chat_response = response.json()
print(f"Response: {chat_response['response']}")
```

### JavaScript

```javascript
const BASE_URL = 'http://localhost:8000';

// Analyze emotion
fetch(`${BASE_URL}/api/v1/emotion/text?text=I'm%20happy&user_id=user123`, {
  method: 'POST'
})
  .then(res => res.json())
  .then(data => console.log('Emotion:', data.dominant_emotion));

// Chat
fetch(`${BASE_URL}/api/v1/chat`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: 'Hello!',
    user_id: 'user123'
  })
})
  .then(res => res.json())
  .then(data => console.log('Response:', data.response));
```

---

## Support

For API support:
- GitHub Issues: https://github.com/yourusername/emotion-ai/issues
- Documentation: https://docs.yourdomain.com
- Email: api-support@yourdomain.com

---

**Last Updated:** 2024-01-15
**API Version:** 2.0.0
