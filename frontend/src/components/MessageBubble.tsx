import ReactMarkdown from "react-markdown";
import { IconShieldCheck } from "@tabler/icons-react";

export type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  /** When true the content is rendered as markdown (final_report). */
  markdown?: boolean;
};

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  const { role, content, markdown } = message;
  const isUser = role === "user";
  const isReport = !isUser && markdown === true;

  return (
    <div className={`msg-row${isUser ? " user" : ""}`}>
      <div className={`msg-avatar${isUser ? " user" : " system"}`}>
        {isUser ? "H" : <IconShieldCheck size={12} stroke={2} />}
      </div>
      <div
        className={`msg-bubble${isUser ? " user" : " system"}${isReport ? " report" : ""}`}
      >
        {isReport ? (
          <div className="report-markdown">
            <ReactMarkdown
              components={{
                // Open links in a new tab
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        ) : (
          content
        )}
      </div>
    </div>
  );
}
