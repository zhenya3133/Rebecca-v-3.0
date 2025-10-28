import React, { useCallback, useEffect, useRef, useState } from "react";
import { RebeccaCoreService } from "../services/RebeccaCoreService";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type ChatPanelProps = {
  token: string;
  onAvatarSignal?: (payload: { action: string; data?: unknown }) => void;
};

export const ChatPanel: React.FC<ChatPanelProps> = ({ token, onAvatarSignal }) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const createSession = async () => {
      if (!token) return;
      try {
        const session = await RebeccaCoreService.startChatSession(token);
        setSessionId(session.session_id);
      } catch (error) {
        console.error(error);
        setStatus("Failed to start session");
      }
    };
    createSession();
  }, [token]);

  const connectWebSocket = useCallback(
    (id: string) => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      const url = RebeccaCoreService.buildWebSocketURL(id);
      const ws = new WebSocket(url);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: (data.role ?? "assistant") as "assistant",
            content: data.content,
          },
        ]);
        onAvatarSignal?.({ action: "speak", data: data.content });
      };
      ws.onclose = () => {
        setStatus("WebSocket disconnected");
      };
      ws.onerror = () => {
        setStatus("WebSocket error");
      };
      wsRef.current = ws;
    },
    [onAvatarSignal],
  );

  useEffect(() => {
    if (sessionId) {
      connectWebSocket(sessionId);
    }
  }, [sessionId, connectWebSocket]);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || !sessionId) return;
    const message: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: input,
    };
    setMessages((prev) => [...prev, message]);
    setInput("");
    onAvatarSignal?.({ action: "listen" });
    try {
      await RebeccaCoreService.sendChatMessage(token, sessionId, message.content);
    } catch (error) {
      console.error(error);
      setStatus("Failed to send message");
    }
  }, [input, sessionId, token, onAvatarSignal]);

  const handleVoice = useCallback(async () => {
    if (!sessionId) return;
    setIsRecording((prev) => !prev);
    onAvatarSignal?.({ action: "voice-record", data: { active: !isRecording } });
    try {
      const stt = await RebeccaCoreService.transcribeVoice(token, sessionId, "mock");
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: stt.text },
      ]);
      onAvatarSignal?.({ action: "listen" });
      const tts = await RebeccaCoreService.textToSpeech(token, sessionId, stt.text);
      onAvatarSignal?.({ action: "speak", data: tts });
      setStatus(`Voice response ready (${tts.format})`);
    } catch (error) {
      console.error(error);
      setStatus("Voice flow failed");
    }
  }, [sessionId, token, isRecording, onAvatarSignal]);

  return (
    <section className="chat-panel">
      <header>
        <h3>Chat</h3>
        {status && <span className="status-text">{status}</span>}
      </header>
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.role}`}>
            <span>{msg.content}</span>
          </div>
        ))}
      </div>
      <footer>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Type a message"
        />
        <button onClick={sendMessage} disabled={!sessionId}>
          Send
        </button>
        <button onClick={handleVoice} disabled={!sessionId}>
          {isRecording ? "Stop" : "Voice"}
        </button>
      </footer>
    </section>
  );
};
