"use strict";

const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { infrastructureChecksRequired, g1SmokeRequired, changedPaths, writeG1Scope } =
  require("../scripts/agent/ci_scope.js");

function setup(eventName = "push", pullRequests = []) {
  const calls = [], messages = [];
  return {
    calls, messages,
    context: { eventName, ref: "refs/heads/codex/infra", repo: { owner: "owner", repo: "repo" } },
    core: { info: (message) => messages.push(message), warning: (message) => messages.push(message) },
    github: { rest: { pulls: { list: async (args) => { calls.push(args); return { data: pullRequests }; } } } },
  };
}

const openPR = {
  number: 12, state: "open", base: { ref: "main" },
  head: { ref: "codex/infra", repo: { full_name: "owner/repo" } },
};

test("first task push runs full checks so automatic draft-PR creation remains available", async () => {
  const fixture = setup();
  assert.equal(await infrastructureChecksRequired(fixture), true);
  assert.deepEqual(fixture.calls[0], {
    owner: "owner", repo: "repo", head: "owner:codex/infra", base: "main", state: "open", per_page: 100,
  });
});

test("an exact open PR suppresses only the duplicate push checks", async () => {
  assert.equal(await infrastructureChecksRequired(setup("push", [openPR])), false);
  const fixture = setup("pull_request", [openPR]);
  assert.equal(await infrastructureChecksRequired(fixture), true);
  assert.equal(fixture.calls.length, 0);
});

test("closed, wrong-head, wrong-base and fork PRs do not suppress push checks", async () => {
  const candidates = [
    { ...openPR, state: "closed" },
    { ...openPR, head: { ...openPR.head, ref: "codex/other" } },
    { ...openPR, base: { ref: "release" } },
    { ...openPR, head: { ...openPR.head, repo: { full_name: "fork/repo" } } },
    { ...openPR, head: { ...openPR.head, repo: null } },
  ];
  assert.equal(await infrastructureChecksRequired(setup("push", candidates)), true);
});

test("unavailable PR lookup retains full checks", async () => {
  const fixture = setup();
  fixture.github.rest.pulls.list = async () => { throw new Error("API unavailable"); };
  assert.equal(await infrastructureChecksRequired(fixture), true);
  assert.match(fixture.messages[0], /running full push checks/);
});

test("G1 smoke scopes actual inputs, including its own workflow and selector", () => {
  for (const input of [
    ".github/workflows/g1-mujoco-smoke.yml", "scripts/run_g1_mujoco_smoke.sh",
    "scripts/agent/ci_scope.js", "tests/ci_scope.test.js", "third_party/LOCK.md",
    "simulation/mujoco/requirements-smoke.txt", "simulation/mujoco/tests/test_g1_headless_smoke.py",
    "third_party/unitree_mujoco", "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml",
  ]) assert.equal(g1SmokeRequired([input]), true, input);
});

test("docs, hooks and X2 experiments do not invoke unrelated G1 physics", () => {
  assert.equal(g1SmokeRequired([
    "docs/setup/codex.md", ".codex/hooks.json", "scripts/research/issue46/grasp/run_review.sh",
    "simulation/mujoco/models/agibot_x2/loader.py",
  ]), false);
});

test("real git diff preserves deleted and renamed dependency paths and CLI writes scope", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "robotsim-ci-scope-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const git = (...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  git("init", "-q");
  git("config", "user.email", "ci-scope@example.invalid");
  git("config", "user.name", "CI scope test");
  fs.mkdirSync(path.join(root, "simulation/mujoco/tests"), { recursive: true });
  fs.writeFileSync(path.join(root, "simulation/mujoco/tests/old.py"), "fixture\n");
  git("add", "."); git("commit", "-qm", "base");
  const base = git("rev-parse", "HEAD");
  git("mv", "simulation/mujoco/tests/old.py", "renamed.txt");
  git("commit", "-qm", "rename");
  const head = git("rev-parse", "HEAD");
  const paths = changedPaths(base, head, root);
  assert.ok(paths.includes("simulation/mujoco/tests/old.py"));
  assert.equal(g1SmokeRequired(paths), true);
  const output = path.join(root, "output");
  execFileSync(process.execPath, [path.resolve(__dirname, "../scripts/agent/ci_scope.js"), "g1"], {
    cwd: root, env: { ...process.env, EVENT_NAME: "pull_request", PR_BASE_SHA: base,
      PR_HEAD_SHA: head, GITHUB_OUTPUT: output },
  });
  assert.equal(fs.readFileSync(output, "utf8"), "run_smoke=true\n");
  assert.equal(writeG1Scope({ EVENT_NAME: "pull_request", PR_BASE_SHA: head, PR_HEAD_SHA: head,
    GITHUB_OUTPUT: output }, root), false);
});

test("main/manual always run; invalid or unavailable PR revisions cannot silently skip", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "robotsim-ci-scope-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const output = path.join(root, "output");
  for (const event of ["push", "workflow_dispatch"]) {
    assert.equal(writeG1Scope({ EVENT_NAME: event, GITHUB_OUTPUT: output }, root), true);
  }
  assert.throws(() => changedPaths("HEAD", "HEAD", root), /immutable/);
  assert.throws(() => writeG1Scope({ EVENT_NAME: "pull_request", PR_BASE_SHA: "a".repeat(40),
    PR_HEAD_SHA: "b".repeat(40), GITHUB_OUTPUT: output }, root));
});
