import { env } from "@/env.mjs";

/* -------- Base relativa a /api -------- */
const BASE = env.NEXT_PUBLIC_API_URL ?? "/backend";

/* --------- Tipos --------- */
export interface PredictBody {
  year: number;
  round: number;
  driverId: number;
  constructorId: number;
}

export interface PredictResponse {
  driver: string;
  year: number;
  round: number;
  win_probability: number;
  predicted_winner: boolean;
}

export interface WinnerResponse {
  gp: { year: number; round: number };
  top: { driverId: number; driver: string; win_probability: number }[];
}

/* --------- Llamadas --------- */
export async function predict(body: PredictBody) {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PredictResponse>;
}

export async function predictWinner(year: number, round: number, topK = 3) {
  const res = await fetch(
    `${BASE}/predict_winner?year=${year}&round=${round}&top_k=${topK}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<WinnerResponse>;
}

