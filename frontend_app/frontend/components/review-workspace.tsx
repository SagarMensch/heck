"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  Expand,
  Pencil,
  ScanSearch,
  ShieldAlert,
  Clock,
  Crop
} from "lucide-react";
import { DocumentPreview } from "@/components/document-preview";

type ReviewField = {
  name: string;
  value: string;
  confidence: number;
  status: "Verified" | "Review Needed";
  anchor: string;
  editable: boolean;
  groundTruth?: string;
  auditHistory?: { time: string; action: string }[];
};

const emptyField: ReviewField = {
  name: "No Data", value: "Upload and process documents first",
  confidence: 0, status: "Review Needed", anchor: "empty", editable: false,
};

export function ReviewWorkspace() {
  const [fields, setFields] = useState<ReviewField[]>([emptyField]);
  const [selectedField, setSelectedField] = useState("empty");
  
  // New State variables for Techathon Features
  const [role, setRole] = useState<"Maker" | "Checker">("Maker");
  const [evalMode, setEvalMode] = useState(false);
  const [showAudit, setShowAudit] = useState(true);
  
  // Batch & Form Selection
  const [formList, setFormList] = useState<{id: string}[]>([]);
  const [currentFormId, setCurrentFormId] = useState<string>("");
  const [pdfUrl, setPdfUrl] = useState<string>("");

  // Fetch all processed forms on mount
  useEffect(() => {
    fetch("http://localhost:8000/api/results")
      .then(r => r.json())
      .then(data => {
        const results = data.results || {};
        const ids = Object.keys(results);
        if (ids.length > 0) {
          setFormList(ids.map(id => ({ id })));
          setCurrentFormId(ids[0]);
        }
      })
      .catch(() => {});
  }, []);

  // Fetch fields & PDF when selected form changes
  useEffect(() => {
    if (!currentFormId) return;
    fetch(`http://localhost:8000/api/review/${currentFormId}`)
      .then(r => r.json())
      .then(data => {
        if (data?.fields?.length) {
          setFields(data.fields);
          setSelectedField(data.fields[0]?.anchor ?? "empty");
          setPdfUrl(data.pdf_url || "");
        }
      })
      .catch(() => {});
  }, [currentFormId]);

  const summary = useMemo(() => {
    const verified = fields.filter((field) => field.status === "Verified").length;
    const flagged = fields.length - verified;

    return {
      total: fields.length,
      verified,
      flagged,
      completion: `${Math.round((verified / fields.length) * 100)}%`
    };
  }, [fields]);

  const activeFieldData = fields.find(f => f.anchor === selectedField);

  // Diff helper
  const renderValueDiff = (value: string, truth?: string) => {
    if (!evalMode || !truth) return value;
    if (value === truth) return <span style={{color: "var(--success-text)"}}>{value} ✓</span>;
    return (
      <span style={{ display: "flex", flexDirection: "column" }}>
        <span style={{color: "var(--alert-text)", textDecoration: "line-through"}}>{value}</span>
        <span style={{color: "var(--success-text)"}}>{truth}</span>
      </span>
    );
  };

  return (
    <div className="reviewPage">
      <div className="reviewHeader">
        <div>
          <div className="eyebrow">Human-in-the-Loop Review</div>
          <h1>Extracted Data Review Dashboard</h1>
          <p>
            Operator-ready split view with editable fields, confidence badges, and
            validation state aligned to the PRD workflow.
          </p>
        </div>

        <div className="reviewSummary">
          <div>
            <strong>{summary.total}</strong>
            <span>Total Fields</span>
          </div>
          <div>
            <strong>{summary.verified}</strong>
            <span>Verified</span>
          </div>
          <div>
            <strong>{summary.flagged}</strong>
            <span>Review Needed</span>
          </div>
          <div>
            <strong>{summary.completion}</strong>
            <span>Completion</span>
          </div>
        </div>
      </div>

      <div className="reviewLayout">
        <section className="surface documentSurface">
          <div className="documentToolbar">
            <div className="toolbarGroup">
              <label style={{ fontSize: "0.85rem", color: "var(--muted)", fontWeight: "bold" }}>Review Document:</label>
              <select 
                value={currentFormId} 
                onChange={(e) => setCurrentFormId(e.target.value)}
                style={{
                  background: "var(--background)",
                  color: "var(--text)",
                  border: "1px solid var(--surface-border)",
                  borderRadius: "6px",
                  padding: "4px 8px",
                  fontSize: "0.9rem",
                  marginLeft: "8px"
                }}
              >
                {formList.length === 0 && <option value="">No forms available</option>}
                {formList.map(f => (
                  <option key={f.id} value={f.id}>{f.id}</option>
                ))}
              </select>
            </div>

            <div className="toolbarGroup">
              <button className="ghostButton" type="button" style={{padding: "6px 12px", minHeight: "34px", fontSize: "0.85rem", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-pill)", background: "transparent", color: "var(--text)"}}>
                <Crop size={14} style={{marginRight: "4px"}} />
                Draw Region
              </button>
              <button className="iconButton soft" type="button" aria-label="Search bounding boxes">
                <ScanSearch size={16} />
              </button>
              <button className="iconButton soft" type="button" aria-label="Download">
                <Download size={16} />
              </button>
              <button className="iconButton soft" type="button" aria-label="Expand">
                <Expand size={16} />
              </button>
            </div>
          </div>

          <div className="documentViewer">
            <DocumentPreview selectedAnchor={selectedField as any} pdfUrl={pdfUrl} />
          </div>
        </section>

        <section className="surface reviewPanel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px" }}>
          <div className="reviewPanelHeader" style={{ flexWrap: "wrap", borderBottom: "1px solid var(--surface-border)", paddingBottom: "16px", marginBottom: "8px" }}>
            <div>
              <div className="miniTitle">Extracted Fields</div>
              <div className="miniSubtitle">Editable values with confidence traceability</div>
            </div>
            
            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              {/* Feature: Techathon Eval Toggle */}
              <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.85rem", color: "var(--text)", cursor: "pointer", fontWeight: 700, padding: "6px 12px", background: evalMode ? "var(--brand-soft)" : "transparent", borderRadius: "var(--radius-pill)", border: "1px solid var(--surface-border)" }}>
                <input type="checkbox" checked={evalMode} onChange={e => setEvalMode(e.target.checked)} style={{margin: 0}} />
                Eval Mode (Diff)
              </label>

              {/* Feature: Role Switcher */}
              <select 
                value={role} 
                onChange={e => setRole(e.target.value as any)}
                style={{ padding: "6px 12px", borderRadius: "var(--radius-pill)", border: "1px solid var(--surface-border)", background: "transparent", fontWeight: 700, fontSize: "0.85rem", color: "var(--text)" }}
              >
                <option value="Maker">Role: Operator</option>
                <option value="Checker">Role: Supervisor</option>
              </select>

              <button 
                className="primaryButton" 
                type="button" 
                disabled={role === "Maker"}
                style={{ opacity: role === "Maker" ? 0.5 : 1, cursor: role === "Maker" ? "not-allowed" : "pointer" }}
                title={role === "Maker" ? "Only Supervisors can export" : "Export JSON"}
              >
                {role === "Maker" ? <ShieldAlert size={14} /> : null}
                Validate &amp; Export
              </button>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: showAudit ? "2.5fr 1fr" : "1fr", gap: "16px", alignItems: "start" }}>
            <div className="reviewTable" style={{ display: "grid", gap: "8px" }}>
              <div className="reviewRow reviewHead" style={{ display: "grid", gridTemplateColumns: evalMode ? "1fr 1fr 1fr 1fr 1fr" : "1fr 1.5fr 1fr 1fr", padding: "10px 14px", background: "transparent", borderBottom: "1px solid var(--surface-border)" }}>
                <span>Field Name</span>
                <span>Value</span>
                {evalMode && <span>Ground Truth</span>}
                <span>Confidence Score</span>
                <span>Status</span>
              </div>

              {fields.map((field, index) => (
                <div
                  key={field.name}
                  className={`reviewRow reviewData ${selectedField === field.anchor ? "selected" : ""}`}
                  style={{ 
                    display: "grid", 
                    gridTemplateColumns: evalMode ? "1fr 1fr 1fr 1fr 1fr" : "1fr 1.5fr 1fr 1fr", 
                    alignItems: "center",
                    gap: "14px",
                    padding: "10px 14px",
                    borderRadius: "var(--radius-xl)",
                    border: selectedField === field.anchor ? "1px solid var(--brand)" : "1px solid transparent", 
                    background: selectedField === field.anchor ? "var(--brand-soft)" : "var(--neutral-bg)",
                    cursor: "pointer"
                  }}
                  onClick={() => setSelectedField(field.anchor)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedField(field.anchor);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <span style={{ fontWeight: 700, color: "var(--text)" }}>{field.name}</span>
                  <span className="fieldInputWrap" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    {field.editable ? (
                      <>
                        <input
                          value={field.value}
                          onChange={(event) => {
                            const next = [...fields];
                            next[index] = { ...field, value: event.target.value };
                            setFields(next);
                          }}
                          style={{
                            width: "100%", padding: "8px 12px", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-pill)", background: "var(--surface)", color: "var(--text)", fontWeight: 600
                          }}
                        />
                      </>
                    ) : (
                      <span style={{ padding: "8px 0", color: "var(--muted)", fontFamily: "monospace", display: "flex", alignItems: "center", gap: "6px", fontWeight: 600 }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                        {field.value}
                      </span>
                    )}
                  </span>
                  
                  {evalMode && (
                     <span style={{ fontFamily: "monospace", fontSize: "0.85rem", fontWeight: 700 }}>
                       {renderValueDiff(field.value, field.groundTruth)}
                     </span>
                  )}

                  <span>
                    <span className="confidenceBadge" style={{ background: field.confidence > 90 ? "var(--success-bg)" : field.confidence > 80 ? "var(--warning-bg)" : "var(--alert-bg)", color: field.confidence > 90 ? "var(--success-text)" : field.confidence > 80 ? "var(--warning-text)" : "var(--alert-text)", borderRadius: "var(--radius-pill)", padding: "4px 10px", fontSize: "0.8rem", fontWeight: 700, border: "none" }}>
                      {field.confidence}%
                    </span>
                  </span>
                  <span>
                    <span
                      className={`statusBadge ${
                        field.status === "Verified" ? "success" : "warning"
                      }`}
                      style={{ borderRadius: "var(--radius-pill)", fontSize: "0.8rem", padding: "4px 10px", border: "none", background: field.status === "Verified" ? "var(--success-bg)" : "var(--warning-bg)", color: field.status === "Verified" ? "var(--success-text)" : "var(--warning-text)" }}
                    >
                      {field.status === "Verified" ? <CheckCircle2 size={14} style={{marginRight: "4px"}} /> : null}
                      {field.status}
                    </span>
                  </span>
                </div>
              ))}
            </div>

            {/* Feature: Forensic Audit Trail */}
            {showAudit && activeFieldData && (
              <div style={{ background: "var(--bg)", borderRadius: "var(--radius-xl)", padding: "20px", border: "1px solid var(--surface-border)" }}>
                <h3 style={{ margin: "0 0 20px 0", fontSize: "1rem", display: "flex", alignItems: "center", gap: "8px", fontWeight: 800 }}>
                  <Clock size={18} color="var(--brand)" />
                  Forensic Audit Trail
                </h3>
                <div style={{ display: "grid", gap: "16px" }}>
                  {activeFieldData.auditHistory?.map((log, i) => (
                    <div key={i} style={{ display: "grid", gridTemplateColumns: "50px 1fr", gap: "12px", fontSize: "0.85rem", position: "relative" }}>
                      {/* Timeline line */}
                      {i !== activeFieldData.auditHistory!.length - 1 && (
                        <div style={{ position: "absolute", left: "25px", top: "20px", bottom: "-16px", width: "1px", background: "var(--surface-border)" }} />
                      )}
                      <span style={{ color: "var(--brand)", fontFamily: "monospace", fontWeight: 700 }}>{log.time}</span>
                      <span style={{ color: "var(--text)", fontWeight: 500, lineHeight: 1.4 }}>{log.action}</span>
                    </div>
                  ))}
                  {(!activeFieldData.auditHistory || activeFieldData.auditHistory.length === 0) && (
                    <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>No logs available for this field.</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
