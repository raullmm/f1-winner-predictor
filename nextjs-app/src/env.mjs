/**
 *  Pequeña capa para importar variables
 *  -> importa con  `import { env } from "@/env.mjs"`
 */
export const env = {
  NEXT_PUBLIC_API_URL:
    process.env.NEXT_PUBLIC_API_URL ?? "http://108.141.82.240",
};

