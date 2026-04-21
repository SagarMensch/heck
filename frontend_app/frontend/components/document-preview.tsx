type Anchor = "name" | "dob" | "pan" | "policy" | "nominee";

const anchorMap: Record<
  Anchor,
  { label: string; top: string; left: string; width: string; tone: "green" | "yellow" | "blue" }
> = {
  name: { label: "Ramesh Kumar", top: "18%", left: "8%", width: "44%", tone: "blue" },
  dob: { label: "20/03/1985", top: "27%", left: "49%", width: "22%", tone: "green" },
  pan: { label: "ARNPK1234D", top: "43%", left: "42%", width: "24%", tone: "yellow" },
  policy: { label: "123456789", top: "22%", left: "58%", width: "19%", tone: "green" },
  nominee: { label: "Seema Kumari", top: "59%", left: "12%", width: "32%", tone: "green" }
};

export function DocumentPreview({ selectedAnchor, pdfUrl }: { selectedAnchor: Anchor, pdfUrl?: string }) {
  if (pdfUrl) {
    return (
      <div className="documentSheet" style={{ padding: 0, overflow: "hidden", display: "flex" }}>
        <iframe src={pdfUrl} width="100%" height="100%" style={{ border: "none", flexGrow: 1 }} />
      </div>
    );
  }

  return (
    <div className="documentSheet">
      <div className="documentHeader">
        <div>
          <div className="documentTitle">LIC Proposal Form</div>
          <div className="documentSubtitle">Extracted Data Review Dashboard</div>
        </div>
        <div className="licBadge">LIC</div>
      </div>

      <div className="documentRow rowWide">
        <span className="docLabel">Name</span>
        <div className="docLine" />
      </div>

      <div className="documentGrid">
        {Array.from({ length: 18 }).map((_, index) => (
          <div key={index} className="gridCell" />
        ))}
      </div>

      <div className="documentChecklist">
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="checkRow">
            <span className="checkBox" />
            <span className="checkLine" />
          </div>
        ))}
      </div>

      {Object.entries(anchorMap).map(([key, box]) => (
        <div
          key={key}
          className={`fieldHighlight tone-${box.tone} ${
            selectedAnchor === key ? "fieldHighlightActive" : ""
          }`}
          style={{ top: box.top, left: box.left, width: box.width }}
        >
          {box.label}
        </div>
      ))}

      <div className="signatureRow">
        <div className="signatureLine" />
        <div className="signatureLine" />
      </div>
    </div>
  );
}
