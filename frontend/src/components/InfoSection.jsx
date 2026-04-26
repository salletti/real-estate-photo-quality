export default function InfoSection() {
  return (
    <section style={styles.section}>

      <div style={styles.warning}>
        <strong>⚠️ Résultats expérimentaux — POC en cours de développement</strong>
        <p style={styles.warningText}>
          Le dataset d'entraînement est encore limité en taille et en diversité.
          Les scores et suggestions générés peuvent être inexacts ou non représentatifs
          de la qualité réelle d'une photo. Utilisez ces résultats comme indication
          générale, pas comme référence absolue.
        </p>
      </div>

      <h2 style={styles.h2}>Comment fonctionne l'application ?</h2>
      <p style={styles.p}>
        Vous uploadez une photo immobilière et l'application vous retourne un score
        de qualité ainsi que des suggestions d'amélioration concrètes.
      </p>
      <p style={styles.p}>Sous le capot, deux systèmes travaillent ensemble :</p>
      <ul style={styles.ul}>
        <li style={styles.li}>Un réseau de neurones (ResNet18) analyse visuellement la photo et détecte les défauts</li>
        <li style={styles.li}>Un LLM (Groq / LLaMA 3.3 70B) reformule les défauts en suggestions lisibles en français</li>
      </ul>

      <h2 style={styles.h2}>Pipeline actuel</h2>
      <div style={styles.pipeline}>
        <div style={styles.pipelineStep}>
          <span style={styles.pipelineIcon}>🖼️</span>
          <div>
            <strong>Image → ResNet18</strong>
            <p style={styles.pipelineDesc}>
              Le backbone extrait les caractéristiques visuelles : lumière,
              netteté, cadrage, composition.
            </p>
          </div>
        </div>
        <div style={styles.pipelineArrow}>↓</div>
        <div style={styles.pipelineStep}>
          <span style={styles.pipelineIcon}>🔍</span>
          <div>
            <strong>Classifier → 7 défauts</strong>
            <p style={styles.pipelineDesc}>
              Une tête de classification prédit la présence ou l'absence de
              chaque défaut : flou, faible lumière, encombrement, mauvais
              cadrage, inclinaison, mauvaise visibilité, filigrane.
            </p>
          </div>
        </div>
        <div style={styles.pipelineArrow}>↓</div>
        <div style={styles.pipelineStep}>
          <span style={styles.pipelineIcon}>📊</span>
          <div>
            <strong>Scoring → Grade A–F</strong>
            <p style={styles.pipelineDesc}>
              Chaque défaut détecté réduit le score selon sa gravité.
              Le résultat final est une note de 0 à 100.
            </p>
          </div>
        </div>
        <div style={styles.pipelineArrow}>↓</div>
        <div style={styles.pipelineStep}>
          <span style={styles.pipelineIcon}>💬</span>
          <div>
            <strong>LLM (Groq) → Suggestions</strong>
            <p style={styles.pipelineDesc}>
              Les défauts sont transmis à un LLM qui génère des conseils
              concrets et compréhensibles pour un photographe.
            </p>
          </div>
        </div>
      </div>

      <h2 style={styles.h2}>Comment fonctionne le scoring ?</h2>
      <p style={styles.p}>
        Le score final (0–100) est calculé à partir des défauts détectés. Chaque
        défaut abaisse le score selon sa gravité.
      </p>
      <ul style={styles.ul}>
        <li style={styles.li}><strong>A (90–100)</strong> — Excellente qualité, peu ou pas de défauts</li>
        <li style={styles.li}><strong>B (75–89)</strong> — Bonne qualité, quelques points à améliorer</li>
        <li style={styles.li}><strong>C (60–74)</strong> — Qualité correcte, plusieurs défauts visibles</li>
        <li style={styles.li}><strong>D/E (30–59)</strong> — Qualité insuffisante, retravail recommandé</li>
        <li style={styles.li}><strong>F (0–29)</strong> — Photo à refaire</li>
      </ul>

      <h2 style={styles.h2}>Pourquoi 7 défauts et pas plus ?</h2>
      <p style={styles.p}>
        La V1 se concentre sur des défauts <strong>objectifs et mesurables</strong>,
        détectables de manière reproductible sur des photos immobilières :
      </p>
      <ul style={styles.ul}>
        <li style={styles.li}><strong>Flou</strong> — netteté insuffisante</li>
        <li style={styles.li}><strong>Faible luminosité</strong> — exposition trop sombre</li>
        <li style={styles.li}><strong>Encombrement</strong> — espace chargé ou mal rangé</li>
        <li style={styles.li}><strong>Mauvais cadrage</strong> — composition tronquée ou décentrée</li>
        <li style={styles.li}><strong>Inclinaison</strong> — horizon ou verticales non droits</li>
        <li style={styles.li}><strong>Mauvaise visibilité de l'espace</strong> — pièce trop peu visible</li>
        <li style={styles.li}><strong>Filigrane</strong> — logo ou watermark visible</li>
      </ul>
      <p style={styles.p}>
        Un critère comme "composition peu attrayante" a été intentionnellement
        exclu : trop subjectif pour être annoté de manière cohérente, il
        introduirait du bruit dans le dataset plutôt que du signal.
        Ce type de critère esthétique global sera adressé en V2, une fois
        la base de données consolidée.
      </p>

      <h2 style={styles.h2}>Amélioration future — Modélisation contextuelle</h2>
      <div style={styles.future}>
        <p style={styles.p}>
          Une amélioration envisagée pour une V2 consiste à intégrer le{" "}
          <strong>type de pièce</strong> directement dans le pipeline ML, pour
          que le modèle raisonne en contexte métier :
        </p>
        <ul style={styles.ul}>
          <li style={styles.li}>
            Une <strong>cuisine encombrée</strong> n'est pas jugée comme un{" "}
            <strong>jardin dense</strong>
          </li>
          <li style={styles.li}>
            Une <strong>salle de bain sombre</strong> n'est pas évaluée comme un{" "}
            <strong>extérieur lumineux</strong>
          </li>
        </ul>
        <p style={styles.p} style={{...styles.p, marginTop: "12px"}}>
          <strong>Architecture envisagée :</strong>
        </p>
        <ul style={styles.ul}>
          <li style={styles.li}>Image → ResNet18 → features visuelles (512)</li>
          <li style={styles.li}>Room type → encodage one-hot (8)</li>
          <li style={styles.li}>Fusion (520) → classifier contextuel</li>
        </ul>
        <p style={styles.p} style={{...styles.p, marginTop: "12px"}}>
          <strong>Pourquoi ce n'est pas activé actuellement :</strong> le dataset
          est encore trop limité pour entraîner ce type de fusion de manière
          fiable. Ajouter de la complexité architecturale sans volume de données
          suffisant risquerait de dégrader les performances de base.
          La priorité actuelle est la robustesse du pipeline fondamental.
        </p>
      </div>

    </section>
  );
}

const styles = {
  section: {
    marginTop: "48px",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  warning: {
    background: "#fff5f5",
    border: "2px solid #dc2626",
    borderRadius: "8px",
    padding: "16px 20px",
    color: "#dc2626",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  warningText: {
    fontSize: "0.9rem",
    lineHeight: "1.6",
    color: "#b91c1c",
    fontWeight: "normal",
  },
  h2: {
    fontSize: "1.1rem",
    fontWeight: "700",
    marginTop: "8px",
    paddingBottom: "4px",
    borderBottom: "1px solid #e5e7eb",
  },
  p: {
    fontSize: "0.9rem",
    lineHeight: "1.7",
    color: "#374151",
  },
  ul: {
    listStyle: "none",
    display: "flex",
    flexDirection: "column",
    gap: "4px",
    paddingLeft: "0",
  },
  li: {
    fontSize: "0.9rem",
    lineHeight: "1.6",
    color: "#374151",
    paddingLeft: "16px",
    position: "relative",
  },
  pipeline: {
    display: "flex",
    flexDirection: "column",
    background: "#f9fafb",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    padding: "16px 20px",
  },
  pipelineStep: {
    display: "flex",
    gap: "12px",
    alignItems: "flex-start",
  },
  pipelineIcon: {
    fontSize: "1.3rem",
    lineHeight: "1.4",
    flexShrink: 0,
  },
  pipelineDesc: {
    fontSize: "0.85rem",
    lineHeight: "1.5",
    color: "#6b7280",
    margin: "2px 0 0 0",
  },
  pipelineArrow: {
    textAlign: "center",
    color: "#9ca3af",
    fontSize: "1.2rem",
    margin: "4px 0",
    paddingLeft: "20px",
  },
  future: {
    background: "#f0f9ff",
    border: "1px solid #bae6fd",
    borderRadius: "8px",
    padding: "16px 20px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
};
