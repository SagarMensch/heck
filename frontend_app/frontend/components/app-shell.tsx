"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { LayoutDashboard, FileText, BarChart2, Settings, HelpCircle, Search, Bell, Cpu } from "lucide-react";

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  const getNavIcon = (label: string) => {
    switch (label) {
      case "Dashboard": return <LayoutDashboard size={18} strokeWidth={1.5} />;
      case "Extract": return <FileText size={18} strokeWidth={1.5} />;
      case "Reports": return <BarChart2 size={18} strokeWidth={1.5} />;
      case "Settings": return <Settings size={18} strokeWidth={1.5} />;
      default: return <FileText size={18} strokeWidth={1.5} />;
    }
  };

  const navItems = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/extract", label: "Extract" },
    { href: "/reports", label: "Reports" },
    { href: "/settings", label: "Settings" }
  ];

  if (pathname === "/") {
    return <>{children}</>;
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandLockup">
            <div className="brandTitle">SequelString AI</div>
            <div className="brandSubtitle">Document Intelligence</div>
          </div>
        </div>

        <nav className="navList" aria-label="Primary navigation">
          {navItems.map(({ href, label }) => {
            const isActive = pathname === href || pathname.startsWith(`${href}/`);

            return (
              <Link
                key={href}
                href={href}
                className={`navItem ${isActive ? "active" : ""}`}
              >
                {getNavIcon(label)}
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="sidebarFooter">
          <Link className="helpButton" href="/docs">
            <HelpCircle size={18} strokeWidth={1.5} />
            <span>Help Docs</span>
          </Link>
          <Link className="helpButton" href="/" style={{ marginTop: "4px", color: "rgba(255,255,255,0.5)" }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
            <span>Sign Out</span>
          </Link>
        </div>
      </aside>

      <div className="contentWrap">
        <header className="topbar">
          <label className="searchBox" aria-label="Search">
            <Search size={16} strokeWidth={1.5} />
            <input defaultValue="" placeholder="Search document or policy number..." />
            <span className="searchShortcut">id</span>
          </label>

          <div className="topbarActions">
            <Link className="primaryButton topbarButton" href="/extract">
              Create New Upload
            </Link>

            <button className="topbarIconButton" type="button" aria-label="Help" style={{padding: "4px"}}>
              <HelpCircle size={18} strokeWidth={1.5} />
            </button>

            <button className="topbarIconButton" type="button" aria-label="Notifications" style={{padding: "4px"}}>
              <Bell size={18} strokeWidth={1.5} />
              <span className="notificationDot" />
            </button>

            <button className="avatarBadge" type="button" aria-label="Profile">
              SA
            </button>

            <button className="assistantButton" type="button" style={{ fontWeight: 600 }}>
              <Cpu size={16} strokeWidth={1.5} />
              <span>AI Assistant</span>
            </button>
          </div>
        </header>

        <main className="pageContent">{children}</main>
      </div>
    </div>
  );
}
