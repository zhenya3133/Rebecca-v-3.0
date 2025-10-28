"""Smoke tests for chat streaming websocket endpoints."""

import importlib

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _fresh_api():
    import api as api_module

    return importlib.reload(api_module)


def test_chat_stream_roundtrip():
    api_module = _fresh_api()
    client = TestClient(api_module.app)
    headers = {"Authorization": f"Bearer {api_module.API_TOKEN}"}
    session_id = client.post("/chat/session", headers=headers).json()["session_id"]

    with client.websocket_connect(f"/chat/stream/{session_id}") as ws:
        ws.send_json({"content": "ping"})
        message = ws.receive_json()
        assert message == {"role": "assistant", "content": "Streaming echo: ping"}

    history = api_module.CHAT_SESSIONS[session_id]["messages"]
    assert any(entry.get("content") == "ping" for entry in history)


def test_chat_stream_missing_session():
    api_module = _fresh_api()
    client = TestClient(api_module.app)

    try:
        with client.websocket_connect("/chat/stream/unknown-session") as ws:
            payload = ws.receive_json()
            assert payload == {"error": "Session not found"}
    except WebSocketDisconnect:
        # Expected after the server closes the connection for unknown session.
        pass
