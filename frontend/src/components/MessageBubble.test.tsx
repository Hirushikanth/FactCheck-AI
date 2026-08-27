import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageBubble } from "./MessageBubble";

describe("MessageBubble", () => {
  it("renders dialogue replies as GitHub-flavored Markdown", () => {
    render(
      <MessageBubble
        message={{
          role: "assistant",
          content: "**Answer**\n\n| Planet | Position |\n| --- | ---: |\n| Earth | 3 |",
        }}
      />,
    );

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "Earth" })).toBeInTheDocument();
  });
});
