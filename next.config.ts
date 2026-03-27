import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  // only need basepath in CI
  ...(process.env.NODE_ENV === 'production' && { basePath: '/yolo11-onnx' }),
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
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    return config;
  },
};

export default nextConfig;
