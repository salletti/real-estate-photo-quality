import { useState } from "react";
import { predictImage } from "./api/predict";
import UploadForm from "./components/UploadForm";
import ResultCard from "./components/ResultCard";
import InfoSection from "./components/InfoSection";
import "./App.css";

export default function App() {
  const [result, setResult] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(file) {
    setLoading(true);
    setError(null);
    setResult(null);
    setImageUrl(URL.createObjectURL(file));

    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Photos Quality</h1>
      <UploadForm onSubmit={handleSubmit} loading={loading} />
      {error && <p className="error">{error}</p>}
      {result && <ResultCard result={result} imageUrl={imageUrl} />}
      <InfoSection />
    </main>
  );
}
