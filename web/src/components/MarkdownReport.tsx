import type { ReactNode } from "react";

interface MarkdownReportProps {
  content: string;
}

interface HeadingBlock {
  kind: "heading";
  level: number;
  text: string;
}

interface ListBlock {
  kind: "list";
  items: Array<{ depth: number; text: string }>;
}

interface ParagraphBlock {
  kind: "paragraph";
  lines: string[];
}

type ReportBlock = HeadingBlock | ListBlock | ParagraphBlock;

function parseBlocks(content: string): ReportBlock[] {
  const blocks: ReportBlock[] = [];
  let paragraphLines: string[] = [];
  let listItems: ListBlock["items"] = [];

  function flushParagraph() {
    if (paragraphLines.length > 0) {
      blocks.push({ kind: "paragraph", lines: paragraphLines });
      paragraphLines = [];
    }
  }

  function flushList() {
    if (listItems.length > 0) {
      blocks.push({ kind: "list", items: listItems });
      listItems = [];
    }
  }

  for (const rawLine of content.split(/\r?\n/)) {
    const heading = rawLine.match(/^(#{1,3})\s+(.+)$/);
    const listItem = rawLine.match(/^(\s*)-\s+(.+)$/);

    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2] });
    } else if (listItem) {
      flushParagraph();
      listItems.push({ depth: Math.min(Math.floor(listItem[1].length / 2), 2), text: listItem[2] });
    } else if (!rawLine.trim()) {
      flushParagraph();
      flushList();
    } else {
      flushList();
      paragraphLines.push(rawLine.trim());
    }
  }

  flushParagraph();
  flushList();
  return blocks;
}

function inlineMarkdown(value: string): ReactNode[] {
  const tokenPattern = /(`[^`]+`|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of value.matchAll(tokenPattern)) {
    const index = match.index ?? cursor;
    if (index > cursor) {
      nodes.push(value.slice(cursor, index));
    }

    const token = match[0];
    if (token.startsWith("`")) {
      nodes.push(<code key={`code-${index}`}>{token.slice(1, -1)}</code>);
    } else {
      const link = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
      nodes.push(
        link ? (
          <a key={`link-${index}`} href={link[2]} target="_blank" rel="noreferrer">
            {link[1]}
          </a>
        ) : (
          token
        ),
      );
    }
    cursor = index + token.length;
  }

  if (cursor < value.length) {
    nodes.push(value.slice(cursor));
  }
  return nodes;
}

export function MarkdownReport({ content }: MarkdownReportProps) {
  const blocks = parseBlocks(content);

  return (
    <div className="markdown-report">
      {blocks.map((block, index) => {
        if (block.kind === "heading") {
          if (block.level === 1) {
            return <h3 key={`heading-${index}`}>{inlineMarkdown(block.text)}</h3>;
          }
          if (block.level === 2) {
            return <h4 key={`heading-${index}`}>{inlineMarkdown(block.text)}</h4>;
          }
          return <h5 key={`heading-${index}`}>{inlineMarkdown(block.text)}</h5>;
        }

        if (block.kind === "list") {
          return (
            <ul key={`list-${index}`}>
              {block.items.map((item, itemIndex) => (
                <li className={`markdown-list-item--${item.depth}`} key={`${item.text}-${itemIndex}`}>
                  {inlineMarkdown(item.text)}
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={`paragraph-${index}`}>
            {block.lines.map((line, lineIndex) => (
              <span key={`${line}-${lineIndex}`}>
                {inlineMarkdown(line)}
                {lineIndex < block.lines.length - 1 && <br />}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}
