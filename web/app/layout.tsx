import type { Metadata } from "next";
import { ThemeProvider } from "@/components/theme-provider";
import CursorTrail from "@/components/cursor-trail";
import "./globals.css";

export const metadata: Metadata = {
  title: "GPT Analytics Dashboard",
  description: "Analyze your ChatGPT conversation data with advanced insights and topic modeling",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background font-mono antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <CursorTrail />
        </ThemeProvider>
      </body>
    </html>
  );
} 