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
      "px-4 py-2 bg-[#5271ff] hover:bg-[#4363e6] rounded text-white w-full disabled:opacity-60 disabled:pointer-events-none":
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





