// TODO: migrate from app.py:920-941
export default function RetrievedChunks({ chunks }: { chunks: unknown[] }) {
  if (!chunks.length) return null;
  return <div className="retrieved-chunks">{/* TODO */}</div>;
}
