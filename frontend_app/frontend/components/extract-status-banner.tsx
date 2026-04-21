"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";

export function ExtractStatusBanner() {
  const searchParams = useSearchParams();
  const uploaded = searchParams.get("uploaded");
  const file = searchParams.get("file");

  if (!uploaded || !file) {
    return null;
  }

  return (
    <div className="statusNotice">
      <div>
        <strong>{file}</strong> selected successfully.
        <span> Continue below to review the extraction queue or open the review workspace.</span>
      </div>
      <Link href="/extract/review" className="primaryButton inlineButton">
        Open Review Screen
      </Link>
    </div>
  );
}
