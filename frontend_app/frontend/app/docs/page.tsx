import { SectionHeader, Surface } from "@/components/dashboard-primitives";

const docSections = [
  {
    title: "Platform Scope",
    text: "The frontend covers the main dashboard, extract workflow, review workspace, reports, and settings for the LIC handwritten proposal form extraction demo."
  },
  {
    title: "Upload & Review Flow",
    text: "Upload actions route operators into the extract flow, and flagged fields can be reviewed in the split-screen workspace with editable values and confidence indicators."
  },
  {
    title: "Metrics & Validation",
    text: "Reports reflect the PRD targets for field accuracy, character accuracy, rejection rate, correction rate, and per-field F1 analysis."
  }
];

export default function DocsPage() {
  return (
    <div className="pageStack">
      <div className="pageIntro">
        <div className="eyebrow">Product Docs</div>
        <h1>LIC IDP Frontend Guide</h1>
        <p>
          Reference documentation for the demo frontend built from the PRD and the
          supplied dashboard/review designs.
        </p>
      </div>

      <div className="settingsGrid">
        {docSections.map((section) => (
          <Surface key={section.title}>
            <SectionHeader title={section.title} />
            <p className="docParagraph">{section.text}</p>
          </Surface>
        ))}
      </div>
    </div>
  );
}
