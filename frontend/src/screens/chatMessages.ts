import type { ChatMessage } from "../components/MessageBubble";

export function appendAssistantMessage(
  messages: ChatMessage[],
  content: string,
): ChatMessage[] {
  if (messages.some((message) => message.role === "assistant" && message.content === content)) {
    return messages;
  }
  return [...messages, { role: "assistant", content }];
}
