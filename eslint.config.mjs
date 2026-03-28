import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const eslintConfig = [...nextCoreWebVitals, ...nextTypescript, {
  ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "next-env.d.ts", ".git_tmp_repo/**", ".git_tmp_repo_fast/**"]
}];

export default eslintConfig;
