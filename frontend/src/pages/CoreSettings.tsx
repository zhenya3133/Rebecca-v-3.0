import React, { useEffect, useState } from "react";
import { UploadDropzone } from "../components/UploadDropzone";
import { ChatPanel } from "../components/ChatPanel";
import { RebeccaCoreService } from "../services/RebeccaCoreService";

type StatusVariant = "idle" | "connected" | "failed" | "error";

export const CoreSettings: React.FC = () => {
  const [token, setToken] = useState("");
  const [endpoint, setEndpoint] = useState("http://localhost:8000");
  const [transport, setTransport] = useState("grpc");
  const [timeoutSeconds, setTimeoutSeconds] = useState<number>(30);
  const [llmDefault, setLlmDefault] = useState("creative");
  const [llmFallback, setLlmFallback] = useState("default");
  const [sttEngine, setSttEngine] = useState("whisper");
  const [ttsEngine, setTtsEngine] = useState("edge");
  const [ingestPipeline, setIngestPipeline] = useState("auto");
  const [status, setStatus] = useState<StatusVariant>("idle");
  const [isTesting, setIsTesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const labelForStatus: Record<StatusVariant, string> = {
    idle: "Awaiting test",
    connected: "Connected",
    failed: "Failed",
    error: "Error",
  };

  useEffect(() => {
    const fetchInitialSettings = async () => {
      if (!token) {
        return;
      }
      try {
        const data = await RebeccaCoreService.fetchSettings(token);
        setEndpoint(data.core.endpoint);
        setTransport(data.core.transport);
        setTimeoutSeconds(data.core.timeout_seconds);
        setLlmDefault(data.llm.default);
        setLlmFallback(data.llm.fallback);
        setSttEngine(data.voice.stt);
        setTtsEngine(data.voice.tts);
        setIngestPipeline(data.documents.ingest_pipeline);
        setLoadError(null);
      } catch (error) {
        console.error(error);
        setLoadError("Failed to load settings");
      }
    };
    fetchInitialSettings();
  }, [token]);

  const handleTestConnection = async () => {
    setIsTesting(true);
    try {
      const result = await RebeccaCoreService.testConnection(endpoint, token);
      setStatus(result ? "connected" : "failed");
    } catch (error) {
      console.error(error);
      setStatus("error");
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async () => {
    if (!token) {
      return;
    }
    setIsSaving(true);
    try {
      await RebeccaCoreService.updateSettings(token, {
        core: {
          endpoint,
          auth_token: token,
          transport,
          timeout_seconds: timeoutSeconds,
        },
        llm: {
          default: llmDefault,
          fallback: llmFallback,
        },
        voice: {
          stt: sttEngine,
          tts: ttsEngine,
        },
        documents: {
          ingest_pipeline: ingestPipeline,
        },
      });
      setLoadError(null);
    } catch (error) {
      console.error(error);
      setLoadError("Failed to save settings");
    } finally {
      setIsSaving(false);
    }
  };

  const handleFileUpload = async (files: FileList) => {
    try {
      await RebeccaCoreService.uploadDocuments(token, files);
      setUploadStatus(`Uploaded ${files.length} file(s)`);
    } catch (error) {
      console.error(error);
      setUploadStatus("Upload failed");
    }
  };

  return (
    <section className="core-settings">
      <header>
        <h2>Rebecca Core Connection</h2>
        <p>Configure endpoint and credentials to link DROId with Rebecca core services.</p>
      </header>
      {loadError && <p className="error">{loadError}</p>}
      <div className="form-group">
        <label htmlFor="core-token">Token</label>
        <input
          id="core-token"
          value={token}
          onChange={(event) => setToken(event.target.value)}
          type="password"
          placeholder="Bearer token"
        />
      </div>
      <div className="form-two-column">
        <div className="form-group">
          <label htmlFor="core-endpoint">Endpoint</label>
          <input
            id="core-endpoint"
            value={endpoint}
            onChange={(event) => setEndpoint(event.target.value)}
            placeholder="http://localhost:8000"
          />
        </div>
        <div className="form-group">
          <label htmlFor="core-transport">Transport</label>
          <input
            id="core-transport"
            value={transport}
            onChange={(event) => setTransport(event.target.value)}
          />
        </div>
      </div>
      <div className="form-two-column">
        <div className="form-group">
          <label htmlFor="core-timeout">Timeout (sec)</label>
          <input
            id="core-timeout"
            type="number"
            value={timeoutSeconds}
            onChange={(event) => setTimeoutSeconds(Number(event.target.value))}
          />
        </div>
        <div className="form-group">
          <label htmlFor="core-ingest">Ingest Pipeline</label>
          <input
            id="core-ingest"
            value={ingestPipeline}
            onChange={(event) => setIngestPipeline(event.target.value)}
          />
        </div>
      </div>
      <div className="form-two-column">
        <div className="form-group">
          <label htmlFor="llm-default">LLM Default</label>
          <input
            id="llm-default"
            value={llmDefault}
            onChange={(event) => setLlmDefault(event.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="llm-fallback">LLM Fallback</label>
          <input
            id="llm-fallback"
            value={llmFallback}
            onChange={(event) => setLlmFallback(event.target.value)}
          />
        </div>
      </div>
      <div className="form-two-column">
        <div className="form-group">
          <label htmlFor="stt-engine">STT Engine</label>
          <input
            id="stt-engine"
            value={sttEngine}
            onChange={(event) => setSttEngine(event.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="tts-engine">TTS Engine</label>
          <input
            id="tts-engine"
            value={ttsEngine}
            onChange={(event) => setTtsEngine(event.target.value)}
          />
        </div>
      </div>
      <div className="actions">
        <button onClick={handleTestConnection} disabled={isTesting || !token}>
          {isTesting ? "Testing..." : "Test Connection"}
        </button>
        <button onClick={handleSave} disabled={isSaving || !token}>
          {isSaving ? "Saving..." : "Save Settings"}
        </button>
        <span className={`status status-${status}`}>{labelForStatus[status]}</span>
      </div>
      <section className="upload-area">
        <h3>Document Upload</h3>
        <UploadDropzone onFilesSelected={handleFileUpload} />
        {uploadStatus && <p className="status-message">{uploadStatus}</p>}
      </section>
      <section className="chat-area">
        <h3>Live Chat</h3>
        <ChatPanel token={token} onAvatarSignal={(payload) => console.log("avatar", payload)} />
      </section>
    </section>
  );
};