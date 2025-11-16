"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app


class TestHealthEndpoints:
    """Test health and status endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'services' in data

    def test_api_status(self, client):
        """Test API status endpoint."""
        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'running'
        assert 'active_sessions' in data


class TestEmotionEndpoints:
    """Test emotion detection endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_text_sentiment_analysis(self, client):
        """Test text sentiment endpoint."""
        response = client.post(
            "/api/v1/emotion/text",
            params={"text": "I'm feeling great today!", "user_id": "test_user"}
        )

        assert response.status_code == 200
        data = response.json()

        assert 'dominant_emotion' in data
        assert 'sentiment' in data
        assert 'confidence' in data

    def test_multimodal_detection(self, client):
        """Test multimodal emotion detection."""
        request_data = {
            "text": "I'm happy!",
            "user_id": "test_user"
        }

        response = client.post(
            "/api/v1/emotion/detect",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        assert 'fused_emotion' in data
        assert 'individual_emotions' in data


class TestChatEndpoints:
    """Test chat endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        request_data = {
            "message": "Hello, how are you?",
            "user_id": "test_user",
            "include_voice": False
        }

        response = client.post(
            "/api/v1/chat",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        assert 'response' in data
        assert 'response_mode' in data
        assert 'confidence' in data
        assert 'timestamp' in data

    def test_chat_with_emotion_context(self, client):
        """Test chat with emotion context."""
        request_data = {
            "message": "I'm feeling stressed",
            "user_id": "test_user",
            "emotion_context": {
                "dominant_emotion": "sad",
                "confidence": 0.8
            }
        }

        response = client.post(
            "/api/v1/chat",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        # Should use supportive mode for sad emotion
        assert data['response_mode'] in ['supportive', 'calm']


class TestUserEndpoints:
    """Test user profile endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_user_stats(self, client):
        """Test get user stats."""
        response = client.get("/api/v1/user/test_user/stats")

        assert response.status_code == 200
        data = response.json()

        assert data['user_id'] == 'test_user'
        assert 'total_interactions' in data
        assert 'mood_patterns' in data
        assert 'interaction_stats' in data

    def test_get_emotion_history(self, client):
        """Test get emotion history."""
        response = client.get("/api/v1/user/test_user/history")

        assert response.status_code == 200
        data = response.json()

        assert data['user_id'] == 'test_user'
        assert 'history' in data
        assert 'count' in data

    def test_set_user_preference(self, client):
        """Test set user preference."""
        response = client.post(
            "/api/v1/user/test_user/preference",
            params={"key": "theme", "value": "dark"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'success'
        assert data['key'] == 'theme'
        assert data['value'] == 'dark'


class TestConfigEndpoints:
    """Test configuration endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_config_info(self, client):
        """Test get config info."""
        response = client.get("/api/v1/config")

        assert response.status_code == 200
        data = response.json()

        assert 'emotion_detection' in data
        assert 'response_modes' in data
        assert 'features' in data


@pytest.mark.integration
class TestWebSocketCommunication:
    """Test WebSocket functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws/test_session") as websocket:
            # Send a chat message
            websocket.send_json({
                "type": "chat",
                "data": {
                    "message": "Hello!",
                    "user_id": "test_user"
                }
            })

            # Receive response
            response = websocket.receive_json()

            assert response['type'] == 'chat_response'
            assert 'data' in response
            assert 'response' in response['data']

    def test_websocket_ping_pong(self, client):
        """Test WebSocket heartbeat."""
        with client.websocket_connect("/ws/test_session") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})

            # Receive pong
            response = websocket.receive_json()

            assert response['type'] == 'pong'
            assert 'timestamp' in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
