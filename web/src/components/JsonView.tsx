interface JsonViewProps {
  title: string;
  value: unknown;
  defaultOpen?: boolean;
}

export function JsonView({ title, value, defaultOpen = false }: JsonViewProps) {
  return (
    <details className="json-view" open={defaultOpen}>
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
