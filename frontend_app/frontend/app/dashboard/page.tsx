"use client";

import { useEffect, useState } from "react";
import { StatCard, FieldConfidenceChart, ProcessingTimeChart, DocBreakdownSection } from "@/components/dashboard-analytics";
import { SectionHeader, Surface, BatchTerminal } from "@/components/dashboard-primitives";
import { fetchDashboard } from "@/lib/api-client";

type DashboardData = {
  stats: { label: string; value: string; detail: string }[];
  uploadJobs: { id: string; customer: string; channel: string; status: string; confidence: string; updatedAt: string }[];
  fieldConfidenceData: { field: string; confidence: number }[];
  logs: string[];
};

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const d = await fetchDashboard();
        if (active) setData(d);
      } catch (e: unknown) {
        if (active) setError(e instanceof Error ? e.message : "Failed to connect to backend");
      }
    };

    load();
    // Auto-refresh every 5 seconds during processing
    const interval = setInterval(load, 5000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  if (error) {
    return (
      <div className="pageStack">
        <Surface>
          <div style={{ padding: "40px", textAlign: "center" }}>
            <h2 style={{ color: "var(--tfl-red)", marginBottom: "8px" }}>Backend Not Connected</h2>
            <p style={{ color: "var(--muted)" }}>Start the API server: <code>python -m src.api</code></p>
            <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginTop: "8px" }}>{error}</p>
          </div>
        </Surface>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="pageStack">
        <Surface>
          <div style={{ padding: "40px", textAlign: "center", color: "var(--muted)" }}>
            Loading pipeline data...
          </div>
        </Surface>
      </div>
    );
  }

  return (
    <div className="pageStack">
      <BatchTerminal />

      {/* ── Top stat cards ── */}
      <div className="metricGrid four">
        {data.stats.map((stat) => (
          <StatCard
            key={stat.label}
            label={stat.label}
            value={stat.value}
            detail={stat.detail}
            accent={stat.label === "Missed Fields"}
          />
        ))}
      </div>

      {/* ── Charts row ── */}
      <div className="analyticsChartsRow">
        <FieldConfidenceChart />
        <ProcessingTimeChart />
      </div>

      {/* ── Document Processing Breakdown ── */}
      <DocBreakdownSection />

      {/* ── Recent Upload Activity ── */}
      <Surface>
        <SectionHeader title="Recent Upload Activity" eyebrow="Job Queue" />
        <div className="dataTable">
          <div className="tableRow tableHead columns6">
            <span>Job ID</span>
            <span>Document</span>
            <span>Channel</span>
            <span>Status</span>
            <span>Confidence</span>
            <span>Processing Time</span>
          </div>
          {data.uploadJobs.length === 0 && (
            <div className="tableRow columns6" style={{ color: "var(--muted)", fontStyle: "italic" }}>
              <span>No documents processed yet. Upload PDFs to begin.</span>
            </div>
          )}
          {data.uploadJobs.map((job) => (
            <div key={job.id} className="tableRow columns6">
              <span>{job.id}</span>
              <span>{job.customer}</span>
              <span>{job.channel}</span>
              <span>
                <span
                  className={`statusBadge ${
                    job.status === "Completed"
                      ? "success"
                      : job.status === "In Review"
                        ? "warning"
                        : "neutral"
                  }`}
                >
                  {job.status}
                </span>
              </span>
              <span>{job.confidence}</span>
              <span>{job.updatedAt}</span>
            </div>
          ))}
        </div>
      </Surface>
    </div>
  );
}
