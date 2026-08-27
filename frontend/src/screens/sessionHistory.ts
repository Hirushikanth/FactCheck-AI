import type { ChatMessage } from "../components/MessageBubble";
import type { DialogueMessage, FactCheckRunSummary } from "../api/types";

type SessionHistory = {
  raw_input: string;
  final_report: string | null;
  runs: Array<Pick<FactCheckRunSummary, "run_id" | "raw_input">>;
  messages: Array<Pick<DialogueMessage, "id" | "role" | "content">>;
};

export function buildHistoricChatMessages(session: SessionHistory): ChatMessage[] {
  const initialRun = session.runs[0];
  const messages: ChatMessage[] = [
    {
      role: "system",
      content: "Hello! Submit a claim and I'll verify it using multiple sources.",
    },
    {
      role: "user",
      content: initialRun?.raw_input ?? session.raw_input,
      activityId: initialRun?.run_id,
      activityKind: "pipeline",
    },
  ];

  if (session.final_report) {
    messages.push({ role: "assistant", content: session.final_report, markdown: true });
  }

  for (const message of session.messages) {
    messages.push({
      role: message.role,
      content: message.content,
      activityId: message.role === "user" ? `dialogue:${message.id}` : undefined,
      activityKind: message.role === "user" ? "dialogue" : undefined,
    });
  }
  return messages;
}
