// TODO: migrate from app.py:889-906
export default function AnswerCard({ answer }: { answer: string | null }) {
  if (!answer) return null;
  return <div className="answer-card">{answer}</div>;
}
