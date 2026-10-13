import { NextRequest, NextResponse } from "next/server";

const SEARCH_SERVICE_URL = process.env.SEARCH_SERVICE_URL ?? "http://search:8000";
const GENERATION_SERVICE_URL = process.env.GENERATION_SERVICE_URL ?? "http://generation:8000";

export async function POST(request: NextRequest) {
  // TODO: orchestrate SEARCH_SERVICE_URL /search -> GENERATION_SERVICE_URL /generate,
  // return {answer, sources, retrieved, quota} to the client.
  return NextResponse.json({ error: "not implemented" }, { status: 501 });
}
