"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const { ensurePullRequest, isTaskBranch, issueFromBranch } = require("../scripts/agent/auto_task_pr.js");

const OWNER = "lzy18001500226";
const REPO = "RobotSim";
const SHA_ONE = "a".repeat(40);
const SHA_TWO = "b".repeat(40);

function setup({
  branch = "issue/36-auto-pr-handoff",
  sha = SHA_ONE,
  refExists = true,
  pullResponses = [[]],
  createError = null,
  sourceRepository = `${OWNER}/${REPO}`,
  targetRepository = `${OWNER}/${REPO}`,
  targetIsFork = false,
  event = "push",
} = {}) {
  const calls = { getRef: [], list: [], getContent: [], create: [] };
  const logs = [];
  let responseIndex = 0;
  let refSha = sha;
  const context = {
    repo: { owner: OWNER, repo: REPO },
    payload: {
      repository: { full_name: targetRepository, fork: targetIsFork },
      workflow_run: {
        event,
        head_branch: branch,
        head_sha: sha,
        head_repository: {
          full_name: sourceRepository,
          fork: sourceRepository !== `${OWNER}/${REPO}`,
        },
      },
    },
  };
  const github = {
    rest: {
      git: {
        getRef: async (args) => {
          calls.getRef.push(args);
          if (!refExists) throw Object.assign(new Error("Not Found"), { status: 404 });
          return { data: { object: { sha: refSha } } };
        },
      },
      pulls: {
        list: async (args) => {
          calls.list.push(args);
          const index = Math.min(responseIndex, pullResponses.length - 1);
          responseIndex += 1;
          return { data: pullResponses[index] };
        },
        create: async (args) => {
          calls.create.push(args);
          if (createError) throw createError;
          return { data: { number: 42 } };
        },
      },
      repos: {
        getContent: async (args) => {
          calls.getContent.push(args);
          return {
            data: {
              type: "file",
              content: Buffer.from("## Summary\n\nDescribe the change.\n").toString("base64"),
            },
          };
        },
      },
    },
  };
  const core = { info: (message) => logs.push(message) };
  return { context, github, core, calls, logs, setRefSha: (nextSha) => { refSha = nextSha; } };
}

test("absent PR creates one from the main template with conservative Issue metadata", async () => {
  const mock = setup();
  const result = await ensurePullRequest(mock);

  assert.deepEqual(result, { state: "created", number: 42 });
  assert.equal(mock.calls.create.length, 1);
  assert.equal(mock.calls.create[0].base, "main");
  assert.equal(mock.calls.create[0].head, "issue/36-auto-pr-handoff");
  assert.equal(mock.calls.list[0].base, "main");
  assert.equal(mock.calls.getContent[0].ref, "main");
  assert.match(mock.calls.create[0].title, /^Issue #36:/);
  assert.match(mock.calls.create[0].body, /Describe the change/);
  assert.match(mock.calls.create[0].body, /Head SHA: `a{40}`/);
  assert.match(mock.calls.create[0].body, /Issue: #36/);
  assert.match(mock.calls.create[0].body, /Review required/);
});

test("an existing open PR is reused without creation", async () => {
  const mock = setup({ pullResponses: [[{ number: 41, state: "open" }]] });
  const result = await ensurePullRequest(mock);
  assert.deepEqual(result, { state: "existing", number: 41 });
  assert.equal(mock.calls.create.length, 0);
  assert.equal(mock.calls.getContent.length, 0);
});

test("repeated pushes to one branch update the same PR and do not create another", async () => {
  const mock = setup({ pullResponses: [[], [{ number: 42, state: "open" }]] });
  const first = await ensurePullRequest(mock);
  mock.context.payload.workflow_run.head_sha = SHA_TWO;
  mock.setRefSha(SHA_TWO);
  const second = await ensurePullRequest(mock);

  assert.deepEqual(first, { state: "created", number: 42 });
  assert.deepEqual(second, { state: "existing", number: 42 });
  assert.equal(mock.calls.create.length, 1);
});

test("a concurrent create race rechecks and accepts the new open PR", async () => {
  const mock = setup({
    pullResponses: [[], [{ number: 43, state: "open" }]],
    createError: Object.assign(new Error("Validation Failed"), { status: 422 }),
  });
  const result = await ensurePullRequest(mock);
  assert.deepEqual(result, { state: "existing", number: 43 });
  assert.equal(mock.calls.list.length, 2);
  assert.equal(mock.calls.create.length, 1);
});

test("wrong branch and non-push workflow runs are rejected before API calls", async (t) => {
  await t.test("wrong branch", async () => {
    const mock = setup({ branch: "docs/cleanup" });
    assert.equal((await ensurePullRequest(mock)).state, "skipped");
    assert.equal(mock.calls.getRef.length, 0);
  });
  await t.test("pull request event", async () => {
    const mock = setup({ event: "pull_request" });
    assert.equal((await ensurePullRequest(mock)).state, "skipped");
    assert.equal(mock.calls.getRef.length, 0);
  });
});

test("fork and noncanonical repository events are rejected before API calls", async (t) => {
  await t.test("fork source", async () => {
    const mock = setup({ sourceRepository: "someone/RobotSim" });
    assert.equal((await ensurePullRequest(mock)).state, "skipped");
    assert.equal(mock.calls.getRef.length, 0);
  });
  await t.test("fork target", async () => {
    const mock = setup({ targetIsFork: true });
    assert.equal((await ensurePullRequest(mock)).state, "skipped");
    assert.equal(mock.calls.getRef.length, 0);
  });
  await t.test("noncanonical target", async () => {
    const mock = setup({ targetRepository: "someone/RobotSim" });
    assert.equal((await ensurePullRequest(mock)).state, "skipped");
    assert.equal(mock.calls.getRef.length, 0);
  });
});

test("malformed Issue-looking branch creates a PR without inferring an Issue", async () => {
  const mock = setup({ branch: "issue/36oops-fix" });
  assert.equal(isTaskBranch("issue/36oops-fix"), true);
  assert.equal(issueFromBranch("issue/36oops-fix"), "");
  const result = await ensurePullRequest(mock);
  assert.equal(result.state, "created");
  assert.match(mock.calls.create[0].title, /^RobotSim task:/);
  assert.match(mock.calls.create[0].body, /Issue: not inferred from branch name/);
  assert.doesNotMatch(mock.calls.create[0].body, /Issue: #36/);
});

test("unsafe or malformed ref components are rejected", () => {
  for (const branch of ["issue/foo/", "issue/foo/.hidden", "codex/foo.lock", "issue/foo..bar"]) {
    assert.equal(isTaskBranch(branch), false, branch);
  }
});

test("a deleted task branch is ignored without creating a PR", async () => {
  const mock = setup({ refExists: false });
  const result = await ensurePullRequest(mock);
  assert.equal(result.state, "skipped");
  assert.equal(mock.calls.list.length, 0);
  assert.equal(mock.calls.create.length, 0);
});

test("a closed or merged PR is not reopened or recreated for a reused branch", async (t) => {
  for (const mergedAt of [null, "2026-01-01T00:00:00Z"]) {
    await t.test(mergedAt ? "merged" : "closed without merge", async () => {
      const mock = setup({
        pullResponses: [[{ number: 39, state: "closed", merged_at: mergedAt }]],
      });
      const result = await ensurePullRequest(mock);
      assert.deepEqual(result, { state: "closed", number: 39 });
      assert.equal(mock.calls.create.length, 0);
    });
  }
});

test("the workflow uses a workflow_run from main and never executes the task branch", () => {
  const workflowPath = path.join(__dirname, "..", ".github", "workflows", "auto-task-pr.yml");
  const infraPath = path.join(__dirname, "..", ".github", "workflows", "agent-infra.yml");
  const workflow = fs.readFileSync(workflowPath, "utf8");
  const infra = fs.readFileSync(infraPath, "utf8");
  const infraName = /^name:\s*(.+)$/m.exec(infra)?.[1];
  assert.ok(infraName);
  assert.match(workflow, /workflow_run:/);
  assert.ok(workflow.includes(`      - ${infraName}`));
  assert.match(workflow, /ref: main/);
  assert.match(workflow, /persist-credentials: false/);
  assert.match(workflow, /contents: read[\s\S]*pull-requests: write/);
  assert.match(workflow, /cancel-in-progress: false/);
  assert.doesNotMatch(workflow, /^  push:/m);
  assert.doesNotMatch(workflow, /^\s+run:/m);
  assert.doesNotMatch(workflow, /ref:.*head_branch/);
  assert.doesNotMatch(workflow, /pulls\.create/);
  assert.match(workflow, /auto_task_pr\.js/);
});
