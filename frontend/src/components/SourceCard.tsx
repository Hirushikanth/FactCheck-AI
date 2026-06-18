import { IconWorld, IconBook, IconBuilding, IconSatellite, IconNews } from "@tabler/icons-react";
import { domainFromUrl, truncate } from "../lib/format";

interface Props {
  url: string;
  excerpt?: string;
  index: number;
}

function sourceIcon(url: string) {
  const domain = domainFromUrl(url).toLowerCase();
  if (domain.includes("nasa") || domain.includes("satellite")) return <IconSatellite size={14} />;
  if (domain.includes("nature") || domain.includes("doi") || domain.includes("journal"))
    return <IconBook size={14} />;
  if (
    domain.includes("nsidc") ||
    domain.includes("noaa") ||
    domain.includes("gov") ||
    domain.includes("edu")
  )
    return <IconBuilding size={14} />;
  if (
    domain.includes("reuters") ||
    domain.includes("bbc") ||
    domain.includes("nyt") ||
    domain.includes("times") ||
    domain.includes("news")
  )
    return <IconNews size={14} />;
  return <IconWorld size={14} />;
}

export function SourceCard({ url, excerpt, index }: Props) {
  const domain = domainFromUrl(url);
  const title = truncate(
    excerpt
      ? excerpt.split(".")[0].trim()
      : domain,
    72
  );

  return (
    <div className="source-card">
      <div className="source-icon">{sourceIcon(url)}</div>
      <div className="source-body">
        <a
          className="source-title"
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          title={url}
        >
          {title}
        </a>
        <div className="source-meta">
          {domain} · Source {index + 1}
        </div>
        {excerpt && (
          <p className="source-excerpt">{truncate(excerpt, 160)}</p>
        )}
      </div>
    </div>
  );
}
