import { NextResponse } from "next/server";

const GENERATION_SERVICE_URL = process.env.GENERATION_SERVICE_URL ?? "http://generation:8000";

export async function GET() {
  // TODO: proxy GENERATION_SERVICE_URL /quota
  return NextResponse.json({ error: "not implemented" }, { status: 501 });
}
