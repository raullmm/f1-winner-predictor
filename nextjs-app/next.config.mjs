/** @type {import("next").NextConfig} */
export const nextConfig = {
  reactStrictMode: true,
  experimental: { appDir: false }, // si sigues en páginas clásicas
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: "http://localhost:8000/:path*", // backend FastAPI
      },
    ];
  },
};

const backendURL = process.env.BACKEND_URL || "http://localhost:8000";

export default {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/backend/:path*",         // todo lo que empiece por /api
        destination: `${backendURL}/:path*`, // se reenvía al backend
      },
    ];
  },
};


