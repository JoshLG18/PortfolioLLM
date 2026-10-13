import { NextResponse } from "next/server";

const INGESTION_SERVICE_URL = process.env.INGESTION_SERVICE_URL ?? "http://ingestion:8000";

export async function GET() {
  // TODO: proxy INGESTION_SERVICE_URL /header
  return NextResponse.json({ error: "not implemented" }, { status: 501 });
}
