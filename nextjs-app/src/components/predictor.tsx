"use client";

import React, { useState } from "react";
import { predictWinner, WinnerResponse } from "@/services/api";

export default function Predictor() {
  const [year]       = useState(new Date().getFullYear()); // fijo por ahora
  const [round]      = useState(11);                       // idem (GP nº 16)
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [result, setResult]       = useState<WinnerResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictWinner(year, round, 3);
      setResult(data);
    } catch (err: unknown) {
      setError((err as Error).message ?? "Error desconocido");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="w-full max-w-md space-y-8">
      <form className="space-y-4" onSubmit={handleSubmit}>
        <h1 className="text-2xl font-semibold">Predictor F1</h1>

        <button type="submit" disabled={loading} className="rounded-lg bg-white p-2 text-sm font-medium text-black transition-all ease-in-out hover:shadow-2xl disabled:bg-orange-900">
          {loading ? "Calculando…" : "Predecir"}
        </button>
      </form>

      {error && <p className="text-red-500">{error}</p>}

      {result && (
        <article className="rounded-xl border border-stone-700 p-4 bg-stone-900">
          <h2 className="text-lg font-medium mb-2">
            Top {result.top.length} favoritos<br />
            GP {result.gp.round} ({result.gp.year})
          </h2>

          <ol className="space-y-1">
            {result.top.map((w, idx) => (
              <li key={w.driverId} className="flex justify-between">
                <span>
                  {idx + 1}. {w.driver}
                </span>
              </li>
            ))}
          </ol>
        </article>
      )}
    </section>
  );
}

