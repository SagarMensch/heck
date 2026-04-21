"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUpRight, FileStack, MoveRight, Upload, WandSparkles } from "lucide-react";
import { useRef, useState, useEffect, type ChangeEvent, type ReactNode } from "react";

export function Surface({
  className = "",
  children
}: {
  className?: string;
  children: ReactNode;
}) {
  return <section className={`surface ${className}`.trim()}>{children}</section>;
}

export function MetricCard({
  label,
  value,
  detail
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metricCard">
      <div className="metricValue">{value}</div>
      <div className="metricLabel">{label}</div>
      <div className="metricDetail">{detail}</div>
    </div>
  );
}

export function UploadHero() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    router.push(`/extract?uploaded=1&file=${encodeURIComponent(file.name)}`);
    event.target.value = "";
  };

  return (
    <Surface className="heroSurface">
      <div className="heroBannerMedia">
        <Image
          src="/hero-banner.png"
          alt="AI Proposal Form Extraction banner"
          fill
          priority
          sizes="(max-width: 1360px) 100vw, 1100px"
          className="heroBannerImage"
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          className="visuallyHiddenInput"
          onChange={handleFileChange}
        />
        <button
          type="button"
          className="heroHotspot heroHotspotUpload"
          aria-label="Upload your LIC document"
          onClick={() => fileInputRef.current?.click()}
        />
        <Link
          href="/docs"
          className="heroHotspot heroHotspotDocs"
          aria-label="Visit docs"
        />
      </div>
    </Surface>
  );
}

export function UploadDropzone() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setSelectedFile(file.name);
    router.push(`/extract?uploaded=1&file=${encodeURIComponent(file.name)}`);
    event.target.value = "";
  };

  return (
    <Surface className="uploadSurface">
      <div className="uploadDropzone">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          className="visuallyHiddenInput"
          onChange={handleFileChange}
        />
        <div className="uploadIcon">
          <Upload size={44} />
        </div>
        <div className="uploadTitle">
          Drag &amp; Drop or{" "}
          <button
            type="button"
            className="browseButton"
            onClick={() => fileInputRef.current?.click()}
          >
            Browse
          </button>{" "}
          to Upload File
        </div>
        <div className="uploadSubtitle">Supported files: pdf, jpeg, png</div>
        {selectedFile ? <div className="uploadSelectedFile">Selected: {selectedFile}</div> : null}
        <button
          className="primaryButton"
          type="button"
          onClick={() => fileInputRef.current?.click()}
        >
          Upload File
        </button>
      </div>
    </Surface>
  );
}

export function SectionHeader({
  title,
  eyebrow,
  action
}: {
  title: string;
  eyebrow?: string;
  action?: ReactNode;
}) {
  return (
    <div className="sectionHeader">
      <div>
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}

export function ActionTile({
  title,
  description,
  href,
  icon
}: {
  title: string;
  description: string;
  href: string;
  icon: "review" | "batch" | "metrics";
}) {
  const Icon = icon === "review" ? WandSparkles : icon === "batch" ? FileStack : ArrowUpRight;

  return (
    <Link href={href} className="actionTile">
      <div className="actionTileIcon">
        <Icon size={20} />
      </div>
      <div className="actionTileBody">
        <div className="actionTileTitle">{title}</div>
        <p>{description}</p>
      </div>
      <MoveRight size={18} />
    </Link>
  );
}

export function BatchTerminal() {
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [totalForms, setTotalForms] = useState(0);
  const [completedForms, setCompletedForms] = useState(0);
  const [activeTab, setActiveTab] = useState<"overview" | "logs">("overview");

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const [logsRes, dashRes] = await Promise.all([
          fetch("http://localhost:8000/api/logs").then(r => r.json()).catch(() => ({ logs: [] })),
          fetch("http://localhost:8000/api/dashboard").then(r => r.json()).catch(() => ({ batch_kpis: {} })),
        ]);
        if (!active) return;
        setLogs(logsRes.logs || []);
        const kpis = dashRes.batch_kpis || {};
        const total = kpis.total_forms || 0;
        const done = kpis.completed || 0;
        setTotalForms(total);
        setCompletedForms(done);
        setProgress(total > 0 ? Math.round((done / total) * 100) : 0);
      } catch { /* backend not running */ }
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  return (
    <div className="surface" style={{ background: "var(--surface)", color: "var(--text)", padding: "20px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", gap: "16px", borderBottom: "1px solid var(--surface-border)", marginBottom: "20px", paddingBottom: "10px" }}>
        <button 
          onClick={() => setActiveTab("overview")} 
          style={{ background: "transparent", border: "none", fontWeight: 800, fontSize: "0.95rem", color: activeTab === "overview" ? "var(--tfl-blue)" : "var(--muted)", cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.05em" }}
        >
          Overview
        </button>
        <button 
          onClick={() => setActiveTab("logs")} 
          style={{ background: "transparent", border: "none", fontWeight: 800, fontSize: "0.95rem", color: activeTab === "logs" ? "var(--tfl-blue)" : "var(--muted)", cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.05em" }}
        >
          System Logs
        </button>
      </div>

      {activeTab === "overview" && (
        <div style={{ fontFamily: "var(--font-body)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "16px", color: "var(--tfl-blue)", fontWeight: 800 }}>
            <span>Batch Processing Pipeline — {totalForms} Forms</span>
            <span>{completedForms} / {totalForms} complete</span>
          </div>
          
          <div style={{ width: "100%", background: "var(--surface-border)", height: "8px", borderRadius: "999px", marginBottom: "16px", overflow: "hidden", flexShrink: 0 }}>
            <div style={{ width: `${progress}%`, background: progress === 100 ? "var(--tfl-green)" : "var(--tfl-blue)", height: "100%", transition: "width 0.4s ease" }} />
          </div>
          <div style={{ color: "var(--muted)", fontSize: "0.9rem", fontWeight: 500 }}>
            {totalForms === 0
              ? "Waiting for documents... Upload PDFs to begin processing."
              : progress === 100
                ? `✓ All ${totalForms} documents processed successfully.`
                : `Processing document ${completedForms + 1} of ${totalForms}...`
            }
          </div>
        </div>
      )}

      {activeTab === "logs" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.82rem", background: "#0f172a", color: "#4ade80", padding: "16px", borderRadius: "8px", height: "140px", overflowY: "auto", fontFamily: "Consolas, 'Courier New', monospace" }}>
          {logs.length === 0 && (
            <div style={{ color: "#64748b" }}>No pipeline logs yet. Start processing to see output.</div>
          )}
          {logs.map((log, index) => (
            <div key={index}>{log}</div>
          ))}
        </div>
      )}
    </div>
  );
}

