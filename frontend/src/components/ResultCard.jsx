const GRADE_COLORS = {
  A: "#22c55e",
  B: "#86efac",
  C: "#f97316",
  D: "#ea580c",
  E: "#ef4444",
  F: "#dc2626",
};

export default function ResultCard({ result, imageUrl }) {
  const { score, grade, issues, suggestions } = result;

  const activeIssues = Object.entries(issues)
    .filter(([, value]) => value === 1)
    .map(([key]) => key.replace(/_/g, " "));

  return (
    <div className="result-card">
      <div className="score-row">
        <span className="score">{score}</span>
        <span className="grade" style={{ backgroundColor: GRADE_COLORS[grade] }}>
          {grade}
        </span>
      </div>

      {activeIssues.length > 0 && (
        <section>
          <h3>Issues detected</h3>
          <ul>
            {activeIssues.map((issue) => (
              <li key={issue}>{issue}</li>
            ))}
          </ul>
        </section>
      )}

      {suggestions && (
        <section>
          <h3>Suggestions</h3>
          <p className="suggestions-text">{suggestions}</p>
        </section>
      )}

      {imageUrl && <img src={imageUrl} alt="Analyzed" className="preview" />}
    </div>
  );
}
