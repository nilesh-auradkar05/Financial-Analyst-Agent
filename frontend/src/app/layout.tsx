import type { Metadata } from "next";
import { connection } from "next/server";
import { IBM_Plex_Mono, IBM_Plex_Sans, Source_Serif_4 } from "next/font/google";
import "./globals.css";

const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex-sans",
});
const serif = Source_Serif_4({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-source-serif",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-mono",
});

export const metadata: Metadata = { title: "Alpha Analyst" };

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  // Opt every page into dynamic rendering so the CSP nonce from proxy.ts is applied.
  await connection();
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable} ${mono.variable}`}>
      <body className="bg-page font-sans text-ink antialiased">{children}</body>
    </html>
  );
}
