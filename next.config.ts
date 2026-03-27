import type { NextConfig } from "next";
import { GITHUB_REPO_NAME } from "./src/utils/paths";

const nextConfig: NextConfig = {
  output: 'export',
  // only need basepath in Github CI for pages deployment
  ...(process.env.NODE_ENV === 'production' && { basePath: `/${GITHUB_REPO_NAME}` }),
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  logging: {
    browserToTerminal: true,
  },
  serverExternalPackages: ["@techstark/opencv-js"],
  turbopack: {
    resolveAlias: {
      fs: {
        browser: 'empty-module',
      },
      path: {
        browser: 'empty-module',
      },
      crypto: {
        browser: 'empty-module',
      },
    },
  },
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };

    return config;
  },
};

export default nextConfig;
