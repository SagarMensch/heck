"use client";

import { useEffect, useState } from "react";
import { SectionHeader, Surface } from "@/components/dashboard-primitives";

type ModelInfo = {
  models: { name: string; type: string; license: string; hosting: string; inference_mode?: string; purpose?: string }[];
  infrastructure: { gpu: string; framework: string };
  compliance: { data_residency: string; no_foreign_api: boolean; no_data_export: boolean };
};

const settingsGroups = [
  {
    title: "Inference Controls",
    items: [
      { label: "Confidence Threshold for HITL", value: "0.85" },
      { label: "VLM Verification Enabled", value: "Yes" },
      { label: "Queue Polling Interval", value: "3 sec" },
    ],
  },
  {
    title: "Validation Rules",
    items: [
      { label: "DOB / Age Range", value: "18 - 100 years" },
      { label: "PAN Regex Enforcement", value: "Enabled" },
      { label: "KYC Fuzzy Match Threshold", value: "0.90" },
    ],
  },
  {
    title: "Audit & Export",
    items: [
      { label: "Immutable Audit Trail", value: "Enabled" },
      { label: "Default Export Format", value: "JSON + CSV" },
      { label: "PII Masking in Logs", value: "Enabled" },
    ],
  },
];

export default function SettingsPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/model-info")
      .then((r) => r.json())
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  return (
    <div className="pageStack">
      <div className="pageIntro">
        <div className="eyebrow">Configuration</div>
        <h1>Extraction Controls &amp; Model Details</h1>
        <p style={{ color: "var(--muted)", fontSize: "0.95rem" }}>
          Pipeline configuration, validation rules, and deployed model information.
        </p>
      </div>

      {/* ── Model Info (required at 12:00 hrs) ── */}
      {modelInfo && (
        <Surface>
          <SectionHeader title="AI / LLM Model Details" eyebrow="Techathon Compliance" />
          <div className="dataTable">
            <div className="tableRow tableHead columns4">
              <span>Model</span>
              <span>Type</span>
              <span>License</span>
              <span>Hosting</span>
            </div>
            {modelInfo.models.map((m) => (
              <div key={m.name} className="tableRow columns4">
                <span style={{ fontWeight: 700 }}>{m.name}</span>
                <span>{m.type}</span>
                <span>{m.license}</span>
                <span>{m.hosting || m.purpose}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: "16px", display: "flex", gap: "24px", fontSize: "0.85rem", color: "var(--muted)" }}>
            <span>🖥️ GPU: <strong style={{ color: "var(--text)" }}>{modelInfo.infrastructure.gpu}</strong></span>
            <span>📍 Data Residency: <strong style={{ color: "var(--text)" }}>{modelInfo.compliance.data_residency}</strong></span>
            <span>🔒 No Foreign API: <strong style={{ color: "var(--tfl-green)" }}>{modelInfo.compliance.no_foreign_api ? "Confirmed" : "No"}</strong></span>
          </div>
        </Surface>
      )}

      <div className="settingsGrid">
        {settingsGroups.map((group) => (
          <Surface key={group.title}>
            <SectionHeader title={group.title} />
            <div className="settingsList">
              {group.items.map((item) => (
                <div key={item.label} className="settingsRow">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </Surface>
        ))}
        
        <Surface>
          <SectionHeader title="Data Privacy (DPDPA 2023)" eyebrow="Compliance Actions" />
          <div style={{ padding: "16px 0" }}>
            <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginBottom: "16px" }}>
              To satisfy the "Zero Data Retention" requirement, you can execute a manual purge of all pipeline 
              results, wiping PII from local memory and clearing the disk cache.
            </p>
            <button 
              onClick={async () => {
                if (!confirm("Are you sure? This will wipe all processed JSON results.")) return;
                try {
                  const res = await fetch("http://localhost:8000/api/compliance/purge", { method: "DELETE" });
                  const data = await res.json();
                  alert(data.message);
                } catch (e) {
                  alert("Failed to connect to backend for purge.");
                }
              }}
              style={{
                background: "var(--tfl-red)",
                color: "white",
                padding: "10px 16px",
                border: "none",
                borderRadius: "6px",
                fontWeight: "bold",
                cursor: "pointer",
                width: "100%"
              }}
            >
              Secure Purge: Wipe All Data
            </button>
          </div>
        </Surface>
      </div>
    </div>
  );
}
