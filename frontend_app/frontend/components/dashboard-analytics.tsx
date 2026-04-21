"use client";

import { useState, useEffect } from "react";
import { Filter, Cuboid, Square } from "lucide-react";
import { Icon3D } from "./icon-3d";

// Local types matching what the API returns
type FieldConfBar = { field: string; confidence: number };
type ProcTimeSample = { date: string; time: number };
type DocBreakdownItem = { label: string; count: number; time: string; color: string };

// Defaults used when API has no data
const defaultFieldConfidence: FieldConfBar[] = [
  { field: "Name", confidence: 0 }, { field: "DOB", confidence: 0 },
  { field: "PAN", confidence: 0 }, { field: "Mobile", confidence: 0 },
  { field: "Address", confidence: 0 },
];
const defaultProcTime: ProcTimeSample[] = [
  { date: "N/A", time: 0 },
];
const defaultDocBreakdown: DocBreakdownItem[] = [
  { label: "Proposal Forms", count: 0, time: "-- sec", color: "#0019a8" },
];

/* â”€â”€â”€ Stat Card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
export function StatCard({
  label,
  value,
  detail,
  accent = false
}: {
  label: string;
  value: string;
  detail?: string;
  accent?: boolean;
}) {
  return (
    <div className={`analyticsStatCard ${accent ? "accent" : ""}`}>
      <div className="analyticsStatValue">{value}</div>
      <div className="analyticsStatLabel">{label}</div>
      {detail && <div className="analyticsStatDetail">{detail}</div>}
    </div>
  );
}

/* â”€â”€â”€ Chart Toggle Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function ChartToggle({ is3D, setIs3D }: { is3D: boolean; setIs3D: (val: boolean) => void }) {
  return (
    <button 
      onClick={() => setIs3D(!is3D)}
      style={{
        display: "flex", alignItems: "center", gap: "8px",
        padding: "6px 14px", borderRadius: "999px",
        background: is3D ? "var(--tfl-blue)" : "var(--brand-soft)",
        border: "none",
        color: is3D ? "#ffffff" : "var(--muted)",
        fontSize: "0.8rem", fontWeight: 800, cursor: "pointer",
        textTransform: "uppercase", letterSpacing: "0.05em",
        transition: "all 0.2s"
      }}
    >
      <span style={{ display: "grid", placeItems: "center", width: "22px", height: "22px", borderRadius: "50%", background: is3D ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.05)" }}>
        {is3D ? <Cuboid size={12} /> : <Square size={12} />}
      </span>
      {is3D ? "3D Mode" : "2D Mode"}
    </button>
  );
}

/* â”€â”€â”€ Field Confidence Bar Chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
export function FieldConfidenceChart() {
  const [is3D, setIs3D] = useState(true);
  const [fieldConfidenceData, setData] = useState<FieldConfBar[]>(defaultFieldConfidence);
  const max = 100;
  const chartH = 180;
  const barW = 36;
  const gap = 28;

  useEffect(() => {
    fetch("http://localhost:8000/api/dashboard")
      .then(r => r.json())
      .then(d => { if (d.fieldConfidenceData?.length) setData(d.fieldConfidenceData); })
      .catch(() => {});
  }, []);

  const cols = fieldConfidenceData.length;
  const svgW = cols * (barW + gap) + gap;

  return (
    <div className="analyticsChartCard" style={is3D ? { 
        background: "linear-gradient(135deg, rgba(255,255,255,0.05), transparent)", 
        backdropFilter: "blur(10px)", border: "1px solid rgba(255,255,255,0.2)",
        boxShadow: "0 20px 40px rgba(0,0,0,0.15)"
      } : {}}>
      <div className="analyticsChartHeader" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span className="analyticsChartTitle" style={is3D ? {color: "#1e3a8a", textShadow: "0 1px 2px rgba(0,0,0,0.1)"} : {}}>Average Field Confidence</span>
        <div style={{ display: "flex", gap: "8px" }}>
          <ChartToggle is3D={is3D} setIs3D={setIs3D} />
          <button className="analyticsChartAction" type="button" aria-label="Filter">
            <Filter size={14} />
          </button>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${svgW} ${chartH + 40}`}
        width="100%"
        preserveAspectRatio="none"
        aria-hidden="true"
        className="analyticsBarSvg"
        style={is3D ? { filter: "drop-shadow(0 10px 10px rgba(0,0,0,0.15))" } : {}}
      >
        {/* Horizontal guide lines */}
        {[100, 95, 90, 85, 80].map((pct) => {
          const y = chartH - (pct / max) * chartH;
          return (
            <g key={pct}>
              <line
                x1={0}
                x2={svgW}
                y1={y}
                y2={y}
                stroke={is3D ? "rgba(226, 232, 244, 0.4)" : "#e2e8f4"}
                strokeWidth="1"
                strokeDasharray="4 3"
              />
              <text x={2} y={y - 4} fontSize="9" fill="#94a3b8">
                {pct}%
              </text>
            </g>
          );
        })}

        {/* Bars */}
        {fieldConfidenceData.map((d, i) => {
          const x = gap + i * (barW + gap);
          const barH = (d.confidence / max) * chartH;
          const y = chartH - barH;
          const tflColors = [
            "#0019a8", // Piccadilly Blue
            "#dc241f", // Central Red
            "#9364cc", // Elizabeth Purple
            "#fa7b05", // Overground Orange
            "#00a0e2"  // Victoria Teal
          ];
          const color = tflColors[i % tflColors.length];

          return (
            <g key={d.field}>
              {is3D ? (
                <>
                  {/* Base plate */}
                  <polygon points={`${x},${chartH} ${x+barW},${chartH} ${x+barW+10},${chartH-10} ${x+10},${chartH-10}`} fill="rgba(0,0,0,0.05)" />
                  {/* Front Face */}
                  <rect x={x} y={y} width={barW} height={barH} fill={color} />
                  {/* Top Face (lighter) */}
                  <polygon points={`${x},${y} ${x+barW},${y} ${x+barW+10},${y-10} ${x+10},${y-10}`} fill={color} opacity={0.6} />
                  {/* Right Face (darker) */}
                  <polygon points={`${x+barW},${y} ${x+barW+10},${y-10} ${x+barW+10},${chartH-10} ${x+barW},${chartH}`} fill={color} opacity={0.8} />
                </>
              ) : (
                <>
                  <rect x={x} y={0} width={barW} height={chartH} rx={6} fill="var(--brand-soft)" />
                  <rect x={x} y={y} width={barW} height={barH} rx={6} fill={color} />
                </>
              )}
              {/* Label */}
              <text
                x={x + barW / 2}
                y={chartH + 24}
                textAnchor="middle"
                fontSize="10"
                fill="var(--muted)"
                fontWeight="bold"
              >
                {d.field}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* â”€â”€â”€ Processing Time Chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
export function ProcessingTimeChart() {
  const [is3D, setIs3D] = useState(true);
  const maxTime = 4;
  const chartH = 140;
  const pts = defaultProcTime;
  const svgW = 320;
  const colW = svgW / Math.max(pts.length, 1);

  const pointCoords = pts.map((p, i) => ({
    x: i * colW + colW / 2,
    y: chartH - (p.time / maxTime) * chartH,
    ...p
  }));

  const polyline = pointCoords.map((c) => `${c.x},${c.y}`).join(" ");

  return (
    <div className="analyticsChartCard" style={is3D ? { 
        background: "linear-gradient(135deg, rgba(255,255,255,0.05), transparent)", 
        backdropFilter: "blur(10px)", border: "1px solid rgba(255,255,255,0.2)",
        boxShadow: "0 20px 40px rgba(0,0,0,0.15)"
      } : {}}>
      <div className="analyticsChartHeader" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span className="analyticsChartTitle" style={is3D ? {color: "#1e3a8a", textShadow: "0 1px 2px rgba(0,0,0,0.1)"} : {}}>Processing Time</span>
          <span className="analyticsChartSub">Last 7 Days</span>
        </div>
        <ChartToggle is3D={is3D} setIs3D={setIs3D} />
      </div>

      <svg
        viewBox={`0 0 ${svgW} ${chartH + 28}`}
        width="100%"
        preserveAspectRatio="none"
        aria-hidden="true"
        style={is3D ? { filter: "drop-shadow(0 8px 12px rgba(59, 130, 246, 0.2))" } : {}}
      >
        {/* Guide lines */}
        {[4, 3, 2, 1].map((v) => {
          const y = chartH - (v / maxTime) * chartH;
          return (
            <g key={v}>
              <line
                x1={0}
                x2={svgW}
                y1={y}
                y2={y}
                stroke={is3D ? "rgba(226, 232, 244, 0.4)" : "#e2e8f4"}
                strokeWidth="1"
                strokeDasharray="4 3"
              />
              <text x={2} y={y - 3} fontSize="9" fill="#94a3b8">
                {v * 25}%
              </text>
            </g>
          );
        })}

        {/* Bar columns */}
        {pointCoords.map((c, i) => {
          const bw = colW * 0.5;
          const bx = c.x - bw/2;
          const bh = chartH - c.y;
          if (is3D) {
            return (
              <g key={`bar-${i}`}>
                {/* 3D Cylinder representation */}
                <ellipse cx={c.x} cy={chartH} rx={bw/2} ry={bw/4} fill="#60a5fa" opacity="0.3" />
                <rect x={bx} y={c.y} width={bw} height={bh} fill="url(#timeBarGrad3D)" opacity="0.9" />
                <ellipse cx={c.x} cy={c.y} rx={bw/2} ry={bw/4} fill="#bfdbfe" />
              </g>
            );
          }
          return (
            <rect
              key={`bar-${i}`}
              x={c.x - colW * 0.3}
              y={c.y}
              width={colW * 0.6}
              height={chartH - c.y}
              rx={4}
              fill="url(#timeBarGrad)"
              opacity="0.7"
            />
          );
        })}

        {/* Line */}
        <polyline
          points={polyline}
          fill="none"
          stroke={is3D ? "#1d4ed8" : "#2563eb"}
          strokeWidth={is3D ? "3" : "2"}
          strokeLinejoin="round"
          style={is3D ? { filter: "drop-shadow(0 4px 4px rgba(37,99,235,0.4))" } : {}}
        />

        {/* Dots */}
        {pointCoords.map((c, i) => (
          <circle key={i} cx={c.x} cy={c.y} r={is3D ? 5 : 4} fill={is3D ? "url(#timeDotGrad)" : "#fff"} stroke={is3D ? "#fff" : "#2563eb"} strokeWidth="2" style={is3D ? { filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.3))" } : {}} />
        ))}

        {/* X labels */}
        {pointCoords.map((c, i) => (
          <text key={i} x={c.x} y={chartH + 16} textAnchor="middle" fontSize="9" fill="#94a3b8">
            {c.date.split(" ")[1]}
          </text>
        ))}

        <defs>
          <linearGradient id="timeBarGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#93c5fd" />
            <stop offset="100%" stopColor="#bfdbfe" />
          </linearGradient>
          <linearGradient id="timeBarGrad3D" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="50%" stopColor="#93c5fd" />
            <stop offset="100%" stopColor="#2563eb" />
          </linearGradient>
          <radialGradient id="timeDotGrad" cx="30%" cy="30%" r="70%">
            <stop offset="0%" stopColor="#fff" />
            <stop offset="100%" stopColor="#3b82f6" />
          </radialGradient>
        </defs>
      </svg>
    </div>
  );
}

/* â”€â”€â”€ Donut Chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function DonutChart({ is3D }: { is3D: boolean }) {
  const total = defaultDocBreakdown.reduce((s, d) => s + d.count, 0);
  const r = 54;
  const cx = 70;
  const cy = 70;
  const strokeW = is3D ? 30 : 22;
  const circ = 2 * Math.PI * r;

  let offset = 0;
  const slices = defaultDocBreakdown.map((d) => {
    const frac = d.count / total;
    const dash = frac * circ;
    const slice = { ...d, dashArray: `${dash} ${circ - dash}`, dashOffset: -offset };
    offset += dash;
    return slice;
  });

  const pct = Math.round((defaultDocBreakdown[0].count / total) * 100);

  return (
    <div style={{ position: "relative", width: "140px", height: "140px", perspective: "400px" }}>
      <svg 
        viewBox="0 0 140 140" 
        width="140" 
        height="140" 
        aria-hidden="true"
        style={is3D ? {
          transform: "rotateX(55deg) rotateZ(-30deg)",
          transformOrigin: "center",
          filter: "drop-shadow(0px 30px 10px rgba(0,0,0,0.3))",
          overflow: "visible"
        } : {}}
      >
        {is3D && (
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(0,0,0,0.1)" strokeWidth={strokeW} transform="translate(0, 10)" />
        )}
        {slices.map((s, i) => (
          <g key={i}>
            {is3D && (
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke={s.color}
                strokeWidth={strokeW}
                strokeDasharray={s.dashArray}
                strokeDashoffset={s.dashOffset}
                transform={`rotate(-90 ${cx} ${cy}) translate(0, 6)`}
                opacity={0.6}
              />
            )}
            <circle
              cx={cx}
              cy={cy}
              r={r}
              fill="none"
              stroke={s.color}
              strokeWidth={strokeW}
              strokeDasharray={s.dashArray}
              strokeDashoffset={s.dashOffset}
              transform={`rotate(-90 ${cx} ${cy})`}
              style={is3D ? { strokeLinecap: "butt" } : {}}
            />
          </g>
        ))}
        {!is3D && (
          <>
            <text x={cx} y={cy - 6} textAnchor="middle" fontSize="18" fontWeight="700" fill="#1e293b">
              {pct}%
            </text>
            <text x={cx} y={cy + 12} textAnchor="middle" fontSize="9" fill="#64748b">
              corcoe doc
            </text>
          </>
        )}
      </svg>

      {/* Floating text for 3D mode */}
      {is3D && (
        <div style={{ position: "absolute", top: "45%", left: "50%", transform: "translate(-50%, -50%)", textAlign: "center", textShadow: "0 2px 4px rgba(255,255,255,0.8)" }}>
          <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "#1e293b" }}>{pct}%</div>
          <div style={{ fontSize: "0.6rem", color: "#64748b", textTransform: "uppercase" }}>corcoe doc</div>
        </div>
      )}
    </div>
  );
}

/* â”€â”€â”€ Document Processing Breakdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
export function DocBreakdownSection() {
  const [is3D, setIs3D] = useState(true);
  const total = defaultDocBreakdown.reduce((s, d) => s + d.count, 0);

  return (
    <div className="analyticsBreakdownCard" style={is3D ? { 
        background: "linear-gradient(135deg, rgba(255,255,255,0.05), transparent)", 
        backdropFilter: "blur(10px)", border: "1px solid rgba(255,255,255,0.2)",
        boxShadow: "0 20px 40px rgba(0,0,0,0.15)",
        transformStyle: "preserve-3d"
      } : {}}>
      <div className="analyticsChartHeader" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span className="analyticsChartTitle" style={is3D ? {color: "#1e3a8a", textShadow: "0 1px 2px rgba(0,0,0,0.1)"} : {}}>Document Processing Breakdown</span>
        <ChartToggle is3D={is3D} setIs3D={setIs3D} />
      </div>

      <div className="analyticsBreakdownBody">
        {/* Table */}
        <div className="analyticsBreakdownTable">
          {defaultDocBreakdown.map((d) => (
            <div key={d.label} className="analyticsBreakdownRow" style={is3D ? {
              background: "rgba(255,255,255,0.6)", padding: "8px 12px", borderRadius: "8px",
              boxShadow: "0 4px 6px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,1)",
              transform: "translateZ(10px)"
            } : {}}>
              <span
                className="analyticsBreakdownDot"
                style={{ background: is3D ? `radial-gradient(circle at 30% 30%, ${d.color}, #000)` : d.color, boxShadow: is3D ? "0 2px 4px rgba(0,0,0,0.2)" : "none" }}
              />
              <span className="analyticsBreakdownLabel">{d.label}</span>
              <span className="analyticsBreakdownCount">
                {d.count.toLocaleString()}
              </span>
              <div className="analyticsBreakdownBar" style={is3D ? { boxShadow: "inset 0 2px 4px rgba(0,0,0,0.1)", background: "rgba(0,0,0,0.05)" } : {}}>
                <div
                  className="analyticsBreakdownBarFill"
                  style={{
                    width: `${(d.count / total) * 100}%`,
                    background: is3D ? `linear-gradient(90deg, ${d.color}, #fff)` : d.color,
                    boxShadow: is3D ? "0 2px 4px rgba(0,0,0,0.2)" : "none"
                  }}
                />
              </div>
              <span className="analyticsBreakdownTime">{d.time}</span>
            </div>
          ))}
        </div>

        {/* Donut */}
        <div className="analyticsDonutWrap">
          <DonutChart is3D={is3D} />
        </div>

        {/* Legend */}
        <div className="analyticsDonutLegend">
          {defaultDocBreakdown.map((d) => (
            <div key={d.label} className="analyticsDonutLegendItem" style={is3D ? { background: "rgba(255,255,255,0.5)", padding: "4px 8px", borderRadius: "6px", boxShadow: "0 2px 4px rgba(0,0,0,0.05)" } : {}}>
              <span
                className="analyticsLegendDot"
                style={{ background: is3D ? `radial-gradient(circle at 30% 30%, ${d.color}, #000)` : d.color }}
              />
              <span style={is3D ? { fontWeight: 600, color: "#1e293b" } : {}}>{d.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
