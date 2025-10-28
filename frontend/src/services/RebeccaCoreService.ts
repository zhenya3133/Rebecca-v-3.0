const DEFAULT_ENDPOINT = "http://localhost:8000";

type HealthResponse = {
  status: string;
  meta?: Record<string, unknown>;
};

type CoreSettingsPayload = {
  core: {
    endpoint: string;
    auth_token: string;
    transport: string;
    timeout_seconds: number;
  };
  llm: {
    default: string;
    fallback: string;
  };
  voice: {
    stt: string;
    tts: string;
  };
  documents: {
    ingest_pipeline: string;
  };
};

export class RebeccaCoreService {
  static async testConnection(endpoint: string, token: string): Promise<boolean> {
    const target = endpoint || DEFAULT_ENDPOINT;
    try {
      const response = await fetch(`${target}/health`, {
        method: "GET",
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (!response.ok) {
        return false;
      }
      const payload = (await response.json()) as HealthResponse;
      return payload.status === "ok";
    } catch (error) {
      console.warn("Rebecca core connection failed", error);
      return false;
    }
  }

  static async fetchSettings(token: string): Promise<CoreSettingsPayload> {
    const response = await fetch(`${DEFAULT_ENDPOINT}/core-settings`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    if (!response.ok) {
      throw new Error("Failed to load core settings");
    }
    return (await response.json()) as CoreSettingsPayload;
  }

  static async updateSettings(token: string, payload: CoreSettingsPayload): Promise<CoreSettingsPayload> {
    const response = await fetch(`${DEFAULT_ENDPOINT}/core-settings`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        endpoint: payload.core.endpoint,
        auth_token: payload.core.auth_token,
        transport: payload.core.transport,
        timeout_seconds: payload.core.timeout_seconds,
        llm_default: payload.llm.default,
        llm_fallback: payload.llm.fallback,
        stt_engine: payload.voice.stt,
        tts_engine: payload.voice.tts,
        ingest_pipeline: payload.documents.ingest_pipeline,
      }),
    });
    if (!response.ok) {
      throw new Error("Failed to update core settings");
    }
    return (await response.json()) as CoreSettingsPayload;
  }

  static async uploadDocuments(token: string, files: FileList): Promise<void> {
    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append("file", file);
    });
    const response = await fetch(`${DEFAULT_ENDPOINT}/documents/upload`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });
    if (!response.ok) {
      throw new Error("Failed to upload documents");
    }
  }

  static async startChatSession(token: string): Promise<{ session_id: string }> {
    const response = await fetch(`${DEFAULT_ENDPOINT}/chat/session`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) throw new Error("Failed to start chat session");
    return response.json();
  }

  static async sendChatMessage(token: string, sessionId: string, content: string): Promise<void> {
    const response = await fetch(`${DEFAULT_ENDPOINT}/chat/message`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ session_id: sessionId, content }),
    });
    if (!response.ok) throw new Error("Failed to send chat message");
  }

  static async transcribeVoice(token: string, sessionId: string, audioBase64: string): Promise<{ text: string }> {
    const response = await fetch(`${DEFAULT_ENDPOINT}/voice/stt`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ session_id: sessionId, audio_base64: audioBase64 }),
    });
    if (!response.ok) throw new Error("STT failed");
    return response.json();
  }

  static async textToSpeech(token: string, sessionId: string, text: string): Promise<{ audio_base64: string; format: string }> {
    const response = await fetch(`${DEFAULT_ENDPOINT}/voice/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ session_id: sessionId, text }),
    });
    if (!response.ok) throw new Error("TTS failed");
    return response.json();
  }

  static buildWebSocketURL(sessionId: string): string {
    const url = new URL(`${DEFAULT_ENDPOINT}/chat/stream/${sessionId}`);
    url.protocol = url.protocol.replace("http", "ws");
    return url.toString();
  }
}