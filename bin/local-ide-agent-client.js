#!/usr/bin/env node

import { act, getMemory, observe, registerClient, sendFeedback } from "../sdk/index.js";

const [command, payloadJson = "{}"] = process.argv.slice(2);
const payload = JSON.parse(payloadJson);

async function main() {
  if (command === "register") {
    console.log(JSON.stringify(await registerClient(payload), null, 2));
    return;
  }
  if (command === "observe") {
    console.log(JSON.stringify(await observe(payload), null, 2));
    return;
  }
  if (command === "act") {
    console.log(JSON.stringify(await act(payload), null, 2));
    return;
  }
  if (command === "feedback") {
    console.log(JSON.stringify(await sendFeedback(payload), null, 2));
    return;
  }
  if (command === "memory") {
    console.log(JSON.stringify(await getMemory(payload.user_id || "default"), null, 2));
    return;
  }

  console.error("Usage: local-ide-agent-client <register|observe|act|feedback|memory> '<json>'");
  process.exitCode = 1;
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
