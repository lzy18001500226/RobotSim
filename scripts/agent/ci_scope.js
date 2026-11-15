"use strict";

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");

// Preserve pre-PR push checks: their successful workflow_run opens the draft PR.
// Once a PR exists, its synchronize event owns the full checks for that push.
async function infrastructureChecksRequired({ github, context, core }) {
  if (context.eventName !== "push") return true;
  const branch = context.ref.replace(/^refs\/heads\//, "");
  const { owner, repo } = context.repo;
  try {
    const response = await github.rest.pulls.list({
      owner, repo, head: `${owner}:${branch}`, base: "main", state: "open", per_page: 100,
    });
    const open = response.data.find((pr) =>
      pr.state === "open" && pr.base.ref === "main" && pr.head.ref === branch &&
      pr.head.repo?.full_name === `${owner}/${repo}`,
    );
    if (open) core.info(`PR #${open.number} owns the full checks; skipping duplicate push checks.`);
    return !open;
  } catch (error) {
    core.warning(`PR lookup unavailable; running full push checks: ${error.message}`);
    return true;
  }
}

const G1_FILES = new Set([
  ".github/workflows/g1-mujoco-smoke.yml",
  "scripts/run_g1_mujoco_smoke.sh",
  "scripts/agent/ci_scope.js",
  "tests/ci_scope.test.js",
  "simulation/mujoco/requirements-smoke.txt",
  "third_party/LOCK.md",
  "third_party/unitree_mujoco",
]);

function g1SmokeRequired(paths) {
  return paths.some((name) => G1_FILES.has(name) ||
    name.startsWith("simulation/mujoco/tests/") ||
    name.startsWith("simulation/mujoco/models/g1/") ||
    name.startsWith("third_party/unitree_mujoco/"));
}

function changedPaths(base, head, cwd) {
  if (![base, head].every((sha) => /^[0-9a-f]{40}$/i.test(sha))) {
    throw new Error("CI scope requires immutable base/head commit SHAs");
  }
  // No rename folding: a renamed/deleted smoke dependency must still run checks.
  const names = execFileSync("git", ["diff", "--no-renames", "--name-only", "-z", base, head],
    { cwd, encoding: "utf8", stdio: "pipe" });
  return names.split("\0").filter(Boolean);
}

function writeG1Scope(env = process.env, cwd = process.cwd()) {
  const required = env.EVENT_NAME !== "pull_request" ||
    g1SmokeRequired(changedPaths(env.PR_BASE_SHA, env.PR_HEAD_SHA, cwd));
  if (!env.GITHUB_OUTPUT) throw new Error("GITHUB_OUTPUT is required");
  fs.appendFileSync(env.GITHUB_OUTPUT, `run_smoke=${required}\n`);
  console.log(required ? "G1 smoke required" : "G1 dependencies unchanged; physics smoke not run");
  return required;
}

if (require.main === module) {
  if (process.argv[2] !== "g1") throw new Error("Usage: node scripts/agent/ci_scope.js g1");
  writeG1Scope();
}

module.exports = { infrastructureChecksRequired, g1SmokeRequired, changedPaths, writeG1Scope };
