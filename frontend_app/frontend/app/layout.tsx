import type { Metadata } from "next";
import { Figtree } from "next/font/google";
import type { ReactNode } from "react";
import { AppShell } from "@/components/app-shell";
import "./globals.css";

const figtree = Figtree({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["300", "400", "500", "600", "700", "800", "900"]
});

export const metadata: Metadata = {
  title: "LIC IDP Frontend",
  description: "Next.js frontend for LIC proposal form extraction and review."
};

export default function RootLayout({
  children
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={figtree.variable}>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
