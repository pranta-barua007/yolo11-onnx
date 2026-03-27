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
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        fs: false,
        path: false,
        os: false,
      };
    }

    return config;
  },
};

export default nextConfig;
