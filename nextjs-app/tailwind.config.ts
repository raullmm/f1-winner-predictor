import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";

/* ---------- Plugin de controles ---------- */
const controlsPlugin = plugin(({ addComponents }) => {
  addComponents({
    ".input": {
      /* Tailwind 3+: se usa @apply dentro de un objeto */
      "@apply px-3 py-2 bg-stone-800 rounded w-full focus:outline-none focus:ring focus:ring-red-500/50":
        {},
    },
    ".select": {
      "@apply px-3 py-2 bg-stone-800 rounded w-full focus:outline-none focus:ring focus:ring-red-500/50":
        {},
    },
    ".btn-primary": {
      "rounded-lg bg-white p-2 text-sm font-medium text-black transition-all ease-in-out hover:shadow-2xl disabled:bg-orange-900":
        {},
    },
  });
});

const config: Config = {
  content: [
    "./src/**/*.{ts,tsx,js,jsx,mdx}",
    "./pages/**/*.{ts,tsx,js,jsx,mdx}",
  ],
  theme: { extend: {} },
  plugins: [controlsPlugin],
};

export default config;





