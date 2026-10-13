import "./globals.css";

export const metadata = {
  title: "Ask Portfolio LLM",
  description: "Ask questions about Josh Le Grice's portfolio, projects, and documents.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
