import type { HealthResponse, ProvidersResponse } from "../api/client";
import { StatusBadge } from "./StatusBadge";

interface ProviderStripProps {
  health: HealthResponse | null;
  providers: ProvidersResponse | null;
  isLoading: boolean;
}

export function ProviderStrip({ health, providers, isLoading }: ProviderStripProps) {
  const healthTone = health?.status === "ok" ? "success" : isLoading ? "neutral" : "warning";

  return (
    <section className="provider-strip" aria-label="Backend provider status">
      <div className="provider-strip__item">
        <span>API</span>
        <StatusBadge label={isLoading ? "checking" : health?.status ?? "unknown"} tone={healthTone} />
      </div>
      <div className="provider-strip__item">
        <span>LLM</span>
        <strong>{providers?.llm ?? "not loaded"}</strong>
      </div>
      <div className="provider-strip__item">
        <span>Embedding</span>
        <strong>{providers?.embedding ?? "not loaded"}</strong>
      </div>
      <div className="provider-strip__item">
        <span>Vector DB</span>
        <strong>{providers?.vector_db ?? "not loaded"}</strong>
      </div>
      <div className="provider-strip__item">
        <span>Web Grounding</span>
        <strong>{providers?.web_search ?? "disabled"}</strong>
      </div>
    </section>
  );
}
