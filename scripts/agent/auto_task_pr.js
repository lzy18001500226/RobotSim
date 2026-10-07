"use strict";

const OWNER = "lzy18001500226";
const REPOSITORY = "RobotSim";
const BASE_BRANCH = "main";
const TASK_BRANCH = /^(?:issue|codex)\/[A-Za-z0-9][A-Za-z0-9._/-]*$/;
const ISSUE_BRANCH = /^issue\/([1-9][0-9]{0,8})(?:$|[-/])/;
const SHA = /^[0-9a-f]{40}$/i;

function isTaskBranch(branch) {
  if (typeof branch !== "string" || !TASK_BRANCH.test(branch) || branch.includes("..")) return false;
  return branch.split("/").every(
    (component) =>
      component.length > 0 &&
      !component.startsWith(".") &&
      !component.endsWith(".") &&
      !component.endsWith(".lock"),
  );
}

function issueFromBranch(branch) {
  return ISSUE_BRANCH.exec(branch)?.[1] || "";
}

function taskTitle(branch, issue) {
  let name = branch.slice(branch.indexOf("/") + 1);
  if (issue) name = name.replace(/^[0-9]+(?:[-/])?/, "");
  name = name.replace(/[._/-]+/g, " ").trim().slice(0, 80) || "implementation";
  return `${issue ? `Issue #${issue}` : "RobotSim task"}: ${name}`;
}

function skip(core, message) {
  core.info(`Skipping automatic PR creation: ${message}`);
  return { state: "skipped", message };
}

async function listPullRequests(github, owner, repo, branch) {
  const result = [];
  for (let page = 1; ; page += 1) {
    const response = await github.rest.pulls.list({
      owner,
      repo,
      head: `${owner}:${branch}`,
      base: BASE_BRANCH,
      state: "all",
      per_page: 100,
      page,
    });
    result.push(...response.data);
    if (response.data.length < 100) return result;
  }
}

function existingPullRequest(pullRequests, core) {
  const open = pullRequests.find((pullRequest) => pullRequest.state === "open");
  if (open) {
    core.info(`PR #${open.number} already tracks this branch; pushed commits update it automatically.`);
    return { state: "existing", number: open.number };
  }
  const closed = pullRequests.find((pullRequest) => pullRequest.state === "closed");
  if (closed) {
    core.info(
      `PR #${closed.number} for this branch is already closed${closed.merged_at ? " or merged" : ""}; ` +
        "it will not be reopened or recreated. Use a fresh task branch for new work.",
    );
    return { state: "closed", number: closed.number };
  }
  return null;
}

async function currentBranchRef(github, owner, repo, branch) {
  try {
    return await github.rest.git.getRef({ owner, repo, ref: `heads/${branch}` });
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
}

async function ensurePullRequest({ github, context, core }) {
  const run = context.payload.workflow_run || {};
  const target = context.payload.repository || {};
  const source = run.head_repository || {};
  const canonical = `${OWNER}/${REPOSITORY}`;

  if (
    context.repo.owner !== OWNER ||
    context.repo.repo !== REPOSITORY ||
    target.full_name !== canonical ||
    target.fork === true ||
    source.full_name !== canonical ||
    source.fork === true
  ) {
    return skip(core, "event is not from the canonical RobotSim repository");
  }
  if (run.event !== "push") return skip(core, "source workflow was not triggered by a branch push");
  if (run.conclusion !== "success") {
    return skip(core, "source workflow did not conclude successfully");
  }

  const branch = run.head_branch;
  if (!isTaskBranch(branch)) return skip(core, "source branch is outside issue/** and codex/**");
  if (typeof run.head_sha !== "string" || !SHA.test(run.head_sha)) {
    return skip(core, "source workflow has no valid head SHA");
  }

  const ref = await currentBranchRef(github, OWNER, REPOSITORY, branch);
  if (!ref) return skip(core, "task branch was deleted before processing");
  if (ref.data.object.sha.toLowerCase() !== run.head_sha.toLowerCase()) {
    return skip(core, "a newer push superseded this completed workflow run");
  }

  const prior = existingPullRequest(
    await listPullRequests(github, OWNER, REPOSITORY, branch),
    core,
  );
  if (prior) return prior;

  const templateResponse = await github.rest.repos.getContent({
    owner: OWNER,
    repo: REPOSITORY,
    path: ".github/pull_request_template.md",
    ref: BASE_BRANCH,
  });
  const templateFile = templateResponse.data;
  if (Array.isArray(templateFile) || templateFile.type !== "file" || !templateFile.content) {
    throw new Error("Could not load the standard PR template from main");
  }

  const issue = issueFromBranch(branch);
  const template = Buffer.from(templateFile.content, "base64").toString("utf8").trim();
  const issueLine = issue ? `- Issue: #${issue}` : "- Issue: not inferred from branch name";
  const body =
    `${template}\n\n---\n\n## Automated task handoff\n\n` +
    `- Branch: \`${branch}\`\n- Head SHA: \`${run.head_sha}\`\n${issueLine}\n\n` +
    "**Review required:** CI results and validation/evidence still require review. " +
    "Architecture, hardware, security, and other maintainer gates remain in force. " +
    "This workflow does not approve, merge, or push commits.";

  const latestRef = await currentBranchRef(github, OWNER, REPOSITORY, branch);
  if (!latestRef) return skip(core, "task branch was deleted before PR creation");
  if (latestRef.data.object.sha.toLowerCase() !== run.head_sha.toLowerCase()) {
    return skip(core, "a newer push superseded this completed workflow run");
  }

  try {
    const created = await github.rest.pulls.create({
      owner: OWNER,
      repo: REPOSITORY,
      title: taskTitle(branch, issue),
      head: branch,
      base: BASE_BRANCH,
      body,
    });
    core.info(`Created PR #${created.data.number} for ${branch} at ${run.head_sha}.`);
    return { state: "created", number: created.data.number };
  } catch (error) {
    if (error.status !== 422) throw error;
    const raced = existingPullRequest(
      await listPullRequests(github, OWNER, REPOSITORY, branch),
      core,
    );
    if (raced) {
      core.info("A PR state changed while creating; no duplicate was added.");
      return raced;
    }
    const latestRef = await currentBranchRef(github, OWNER, REPOSITORY, branch);
    if (!latestRef) return skip(core, "task branch was deleted while creating the PR");
    if (latestRef.data.object.sha.toLowerCase() !== run.head_sha.toLowerCase()) {
      return skip(core, "a newer push superseded this completed workflow run");
    }
    throw error;
  }
}

module.exports = {
  ensurePullRequest,
  isTaskBranch,
  issueFromBranch,
};
