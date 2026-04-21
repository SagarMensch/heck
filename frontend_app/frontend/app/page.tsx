import Link from "next/link";
import Image from "next/image";

export default function LandingPage() {
  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: `
        html, body {
          background-color: #000 !important;
          margin: 0 !important;
          padding: 0 !important;
          overflow-x: hidden !important;
        }
      `}} />
      <div style={{ backgroundColor: "#000", color: "#fff", minHeight: "100vh", fontFamily: "var(--font-body), sans-serif", display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>
        {/* Top Nav */}
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "30px 50px", position: "relative", zIndex: 10 }}>
          <div style={{ fontSize: "24px", fontWeight: 700, letterSpacing: "-1px" }}>
            SequelString <span style={{ color: "#00d1d1" }}>AI</span>
          </div>
          <nav style={{ display: "flex", gap: "30px", alignItems: "center", fontSize: "14px", fontWeight: 500 }}>
            <span style={{ cursor: "pointer", opacity: 0.8 }}>Architecture</span>
            <span style={{ cursor: "pointer", opacity: 0.8 }}>Compliance</span>
            <span style={{ cursor: "pointer", opacity: 0.8 }}>Throughput</span>
            <Link href="/dashboard" style={{ border: "1px solid #fff", borderRadius: "999px", padding: "10px 24px", fontWeight: 600, transition: "background 0.2s", cursor: "pointer" }}>
              Enter Dashboard
            </Link>
          </nav>
        </header>

        {/* Main Content */}
        <main style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", padding: "0 80px", position: "relative" }}>
          {/* Left Side: Text */}
          <div style={{ maxWidth: "500px", position: "relative", zIndex: 10 }}>
            <h1 style={{ fontSize: "82px", fontWeight: 500, margin: "0 0 16px 0", letterSpacing: "-3px" }}>
              Form 300
            </h1>
            <p style={{ fontSize: "28px", lineHeight: 1.3, opacity: 0.6, marginBottom: "48px", fontWeight: 300, letterSpacing: "-0.5px" }}>
              Absolute precision.<br/>Infinite scale.
            </p>
            <Link href="/dashboard" style={{ display: "inline-block", fontSize: "15px", textDecoration: "none", color: "#fff", borderBottom: "1px solid #fff", paddingBottom: "4px", fontWeight: 500 }}>
              Enter Dashboard ↓
            </Link>
          </div>

          {/* Right Side: Generated 3D Image */}
          <div style={{ position: "absolute", right: "-100px", top: "50%", transform: "translateY(-50%)", width: "65%", height: "100%", zIndex: 1, opacity: 0.9 }}>
            <img 
              src="/neuralink-core.png" 
              alt="AI Core" 
              style={{ width: "100%", height: "100%", objectFit: "contain", filter: "brightness(0.95) contrast(1.15)" }}
            />
          </div>
        </main>
      </div>
    </>
  );
}
