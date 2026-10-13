import Header from "./components/Header";
import QueryBox from "./components/QueryBox";
import AnswerCard from "./components/AnswerCard";
import SourcesList from "./components/SourcesList";
import RetrievedChunks from "./components/RetrievedChunks";
import QuotaBadge from "./components/QuotaBadge";
import SettingsPanel from "./components/SettingsPanel";

export default function HomePage() {
  // TODO: migrate query flow from app.py:865-971 - submit query, POST /api/ask,
  // render AnswerCard/SourcesList/RetrievedChunks from the response.
  return (
    <main>
      <Header />
      <QuotaBadge />
      <SettingsPanel />
      <QueryBox />
      <AnswerCard answer={null} />
      <SourcesList sources={[]} />
      <RetrievedChunks chunks={[]} />
    </main>
  );
}
