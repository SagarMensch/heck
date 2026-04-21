"use client";

import { useEffect, useState } from "react";
import { MetricCard, SectionHeader, Surface } from "@/components/dashboard-primitives";
import { fetchDashboard } from "@/lib/api-client";

type ReportMetric = { label: string; value: string; change: string };

export default function ReportsPage() {
  const [metrics, setMetrics] = useState<ReportMetric[]>([]);
  const [batchKpis, setBatchKpis] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboard()
      .then((d) => {
        setMetrics(d.reportsMetrics || []);
        setBatchKpis(d.batch_kpis || {});
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="pageStack">
      <div className="pageIntro">
        <div className="eyebrow">Evaluation Dashboard</div>
        <h1>Accuracy, Throughput &amp; Correction Reports</h1>
        <p style={{ color: "var(--muted)", fontSize: "0.95rem" }}>
          Real-time extraction performance, confidence intervals, and automated workflow analytics.
        </p>
      </div>

      {error && (
        <Surface>
          <div style={{ padding: "20px", color: "var(--tfl-red)" }}>
            Backend not connected. Start the API server to see live data.
          </div>
        </Surface>
      )}

      <div className="metricGrid four">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            detail={metric.change}
          />
        ))}
      </div>

      <div className="twoColumnLayout">
        <Surface>
          <SectionHeader title="Batch Performance KPIs" eyebrow="Pipeline Metrics" />
          <div className="dataTable">
            <div className="tableRow tableHead columns4">
              <span>Metric</span>
              <span>Value</span>
              <span>Target</span>
              <span>Status</span>
            </div>
            <div className="tableRow columns4">
              <span>Field-Level Accuracy</span>
              <span>{batchKpis.field_level_accuracy ?? 0}%</span>
              <span>≥ 95%</span>
              <span>
                <span className={`statusBadge ${(batchKpis.field_level_accuracy ?? 0) >= 95 ? "success" : "warning"}`}>
                  {(batchKpis.field_level_accuracy ?? 0) >= 95 ? "PASS" : "BELOW"}
                </span>
              </span>
            </div>
            <div className="tableRow columns4">
              <span>Character Accuracy</span>
              <span>{batchKpis.character_level_accuracy ?? 0}%</span>
              <span>≥ 97%</span>
              <span>
                <span className={`statusBadge ${(batchKpis.character_level_accuracy ?? 0) >= 97 ? "success" : "warning"}`}>
                  {(batchKpis.character_level_accuracy ?? 0) >= 97 ? "PASS" : "BELOW"}
                </span>
              </span>
            </div>
            <div className="tableRow columns4">
              <span>Manual Correction Rate</span>
              <span>{batchKpis.manual_correction_rate ?? 0}%</span>
              <span>≤ 10%</span>
              <span>
                <span className={`statusBadge ${(batchKpis.manual_correction_rate ?? 0) <= 10 ? "success" : "warning"}`}>
                  {(batchKpis.manual_correction_rate ?? 0) <= 10 ? "PASS" : "ABOVE"}
                </span>
              </span>
            </div>
            <div className="tableRow columns4">
              <span>Auto-Rejection Rate</span>
              <span>{batchKpis.auto_rejection_rate ?? 0}%</span>
              <span>≤ 5%</span>
              <span>
                <span className={`statusBadge ${(batchKpis.auto_rejection_rate ?? 0) <= 5 ? "success" : "warning"}`}>
                  {(batchKpis.auto_rejection_rate ?? 0) <= 5 ? "PASS" : "ABOVE"}
                </span>
              </span>
            </div>
            <div className="tableRow columns4">
              <span>Forms Processed</span>
              <span>{batchKpis.completed ?? 0} / {batchKpis.total_forms ?? 0}</span>
              <span>50 / 50</span>
              <span>
                <span className={`statusBadge ${(batchKpis.completed ?? 0) === (batchKpis.total_forms ?? 0) && (batchKpis.total_forms ?? 0) > 0 ? "success" : "neutral"}`}>
                  {(batchKpis.total_forms ?? 0) === 0 ? "WAITING" : (batchKpis.completed ?? 0) === (batchKpis.total_forms ?? 0) ? "DONE" : "IN PROGRESS"}
                </span>
              </span>
            </div>
          </div>
        </Surface>

        <Surface>
          <SectionHeader title="Processing Summary" eyebrow="Batch Statistics" />
          <div className="reportNotes">
            <div className="noteCard">
              <strong>Total Processing Time</strong>
              <p>{batchKpis.total_processing_time_s ?? 0}s for {batchKpis.total_forms ?? 0} forms</p>
            </div>
            <div className="noteCard">
              <strong>Avg Per-Document</strong>
              <p>{((batchKpis.avg_processing_time_ms ?? 0) / 1000).toFixed(1)}s per form</p>
            </div>
            <div className="noteCard">
              <strong>Fields Extracted</strong>
              <p>{batchKpis.extracted_fields ?? 0} of {batchKpis.total_fields ?? 0} total fields</p>
            </div>
            <div className="noteCard">
              <strong>Review Queue</strong>
              <p>{batchKpis.needs_review ?? 0} forms need manual review</p>
            </div>
          </div>
        </Surface>
      </div>
    </div>
  );
}
