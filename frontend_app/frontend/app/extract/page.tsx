"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useRef, useState, type ChangeEvent } from "react";
import { ArrowUpRight, ChevronDown, FileSpreadsheet, Settings2 } from "lucide-react";
import { ReviewWorkspace } from "@/components/review-workspace";

const stats = [
  { value: "98.7%", label: "Avg Recognition Accuracy", detail: "Measured on current demo batch" },
  { value: "2 sec", label: "Avg Processing Time", detail: "Median page turnaround" },
  { value: "80s", label: "Avg Confidence", detail: "Cross-field verification coverage" },
  { value: "20+", label: "Fields Detected", detail: "Per LIC proposal form" }
];

const extractedLeftColumn = [
  { field: "Name", value: "Ramesh Kumar", confidence: "67%", tone: "success" },
  { field: "DOB", value: "20/03/1985", confidence: "96%", tone: "success" },
  { field: "Policy No.", value: "123456789", confidence: "96%", tone: "success" },
  { field: "Nominee Name", value: "Seema Kumari", confidence: "81%", tone: "success" }
];

const extractedRightColumn = [
  { field: "DOB", value: "20/03.31985", confidence: "96%", tone: "success" },
  { field: "PAN", value: "ARNPK1234D", confidence: "96%", tone: "success" },
  { field: "PAN Coer", value: "timvathis. Stracd Kumar", confidence: "96%", tone: "warm" },
  { field: "Nominee Name", value: "Seema Kumari", confidence: "91%", tone: "alert" }
];

/* ── Inner client component that reads search params ── */
function ExtractPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploaded = searchParams.get("uploaded");
  const fileName = searchParams.get("file");

  const [uploading, setUploading] = useState(false);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      // Upload files to backend
      const formData = new FormData();
      for (const file of Array.from(files)) {
        formData.append("files", file);
      }
      const uploadRes = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadRes.json();

      // Start async processing
      await fetch("http://localhost:8000/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: uploadData.folder }),
      });

      // Navigate to review mode
      const firstName = files[0]?.name || "document";
      router.push(`/extract?uploaded=1&file=${encodeURIComponent(firstName)}&count=${files.length}`);
    } catch (e) {
      alert("Upload failed. Is the backend running? (python -m src.api)");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  /* ── When a file has been uploaded → show the Review Dashboard ── */
  if (uploaded === "1") {
    return (
      <div className="extractWorkspace">
        {/* Tab bar stays visible */}
        <div className="extractPageTabs">
          <button className="extractPageTabTrigger" type="button" aria-label="Current module">
            <span>Extract</span>
            <ChevronDown size={16} />
          </button>
          <Link className="extractPageTabLink active" href="/extract">
            Extract
          </Link>
          <Link className="extractPageTabLink" href="/reports">
            <FileSpreadsheet size={15} />
            Reports
          </Link>
          <Link className="extractPageTabLink" href="/settings">
            <Settings2 size={15} />
            Settings
          </Link>

          {/* File name chip + re-upload shortcut */}
          {fileName && (
            <span className="extractUploadedChip">
              📄 {decodeURIComponent(fileName)}
              <button
                type="button"
                className="extractReUploadBtn"
                onClick={() => router.push("/extract")}
              >
                ✕
              </button>
            </span>
          )}
        </div>

        {/* Full review workspace — exactly matches the reference image */}
        <ReviewWorkspace />
      </div>
    );
  }

  /* ── Default: normal extract landing page ── */
  return (
    <div className="extractWorkspace">
      <div className="extractPageTabs">
        <button className="extractPageTabTrigger" type="button" aria-label="Current module">
          <span>Extract</span>
          <ChevronDown size={16} />
        </button>
        <Link className="extractPageTabLink active" href="/extract">
          Extract
        </Link>
        <Link className="extractPageTabLink" href="/reports">
          <FileSpreadsheet size={15} />
          Reports
        </Link>
        <Link className="extractPageTabLink" href="/settings">
          <Settings2 size={15} />
          Settings
        </Link>
      </div>

      <section className="extractHeroCard">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          multiple
          className="visuallyHiddenInput"
          onChange={handleFileChange}
        />

        <div className="extractHeroCopy">
          <h1>AI Proposal Form Extraction</h1>
          <h2>Intelligent OCR for LIC Documents</h2>
          <p>
            Extract key information from scanned LIC proposal forms in seconds with AI.
            Ensure accurate and structured data capture.
          </p>
          <div className="extractHeroActions">
            <button
              className="primaryButton extractCta"
              type="button"
              onClick={() => fileInputRef.current?.click()}
            >
              {uploading ? "Uploading & Processing..." : "Upload LIC Documents"}
            </button>
            <Link className="ghostActionButton" href="/docs">
              Visit Docs
              <ArrowUpRight size={15} />
            </Link>
          </div>
        </div>

        <div className="extractHeroVisual" aria-hidden="true">
          <div className="heroAura heroAuraOne" />
          <div className="heroAura heroAuraTwo" />
          <div className="heroOrbit heroOrbitOne" />
          <div className="heroOrbit heroOrbitTwo" />
          <div className="heroOrbit heroOrbitThree" />

          <div className="heroSheetStack">
            <div className="heroSheet heroSheetBack" />
            <div className="heroSheet heroSheetMid" />
            <div className="heroSheet heroSheetFront">
              <div className="heroSheetHeader" />
              <div className="heroSheetLines">
                {Array.from({ length: 6 }).map((_, index) => (
                  <span
                    key={`sheet-line-${index}`}
                    className={`heroSheetLine ${index === 4 ? "short" : ""}`}
                  />
                ))}
              </div>
              <div className="heroSheetFooter">
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>

          <div className="heroArmBase" />
          <div className="heroArmShoulder">
            <div className="heroArmLink heroArmLinkUpper" />
            <div className="heroArmElbow">
              <div className="heroArmLink heroArmLinkLower" />
              <div className="heroArmWrist">
                <div className="heroArmClaw heroArmClawLeft" />
                <div className="heroArmClaw heroArmClawRight" />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="extractStatStrip">
        {stats.map((stat, index) => (
          <article key={stat.label} className="extractStatCard">
            <div className="extractStatValue">{stat.value}</div>
            <div className="extractStatLabel">{stat.label}</div>
            {index === 0 ? (
              <div className="extractStatSpark" aria-hidden="true">
                {Array.from({ length: 9 }).map((_, sparkIndex) => (
                  <span key={`spark-${sparkIndex}`} />
                ))}
              </div>
            ) : (
              <div className="extractStatDetail">{stat.detail}</div>
            )}
          </article>
        ))}
      </section>

      <section className="extractResultBoard">
        <div className="extractResultGrid">
          <article className="extractRecordCard">
            <div className="extractRecordHeader">Extracted</div>
            <div className="extractFieldList">
              {extractedLeftColumn.map((item) => (
                <div key={`${item.field}-${item.value}`} className="extractFieldRow">
                  <div>
                    <div className="extractFieldLabel">{item.field}</div>
                    <div className="extractFieldValue">{item.value}</div>
                  </div>
                  <span className={`extractConfidenceBadge ${item.tone}`}>{item.confidence}</span>
                </div>
              ))}
            </div>
          </article>

          <article className="extractRecordCard">
            <div className="extractRecordHeader">Validated Output</div>
            <div className="extractFieldList">
              {extractedRightColumn.map((item) => (
                <div key={`${item.field}-${item.value}`} className="extractFieldRow">
                  <div>
                    <div className="extractFieldLabel">{item.field}</div>
                    <div className="extractFieldValue">{item.value}</div>
                  </div>
                  <span className={`extractConfidenceBadge ${item.tone}`}>{item.confidence}</span>
                </div>
              ))}
            </div>
          </article>
        </div>
      </section>
    </div>
  );
}

export default function ExtractPage() {
  return (
    <Suspense fallback={<div className="extractWorkspace" />}>
      <ExtractPageInner />
    </Suspense>
  );
}
